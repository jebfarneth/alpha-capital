from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from alpha.assembly.m3_daily import (
    SHADOW_PATTERN_ID,
    SectorAssignmentSnapshot,
    SectorReturnComponent,
    SectorReturnSnapshot,
    adjusted_return,
    assemble_m3_daily,
    build_sector_return_components,
    compute_sector_return_snapshots,
    nth_previous_session,
)
from alpha.assembly.m3_sector_map import (
    SECTORS,
    SIC_TO_SECTOR_MAP_VERSION,
    canonical_sector_from_fmp,
    major_group_map,
    sector_for_sic,
)
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    FirmSectorAssignment,
    FirmSectorAssignmentHistory,
    SectorReturnDaily,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.m3_daily import M3DailyAssemblyJob, _resolve_delisted_dates_for_missing_end_prices
from alpha.jobs.m3_sector_history import (
    SOURCE_FMP_FALLBACK,
    SOURCE_POLYGON_SIC,
    ResolvedSectorAssignment,
    load_sector_assignments_at,
    resolve_sector_assignment,
    write_sector_assignment_interval,
)
from alpha.jobs.runner import run_job
from alpha.market_calendar import us_equity_session_close_timestamp
from alpha.patterns.m3 import M3Detector


def _ts() -> datetime:
    return datetime(2026, 6, 4, 22, 30, tzinfo=timezone.utc)


def _lineage(provider: str, endpoint: str, payload) -> LineageMeta:
    return LineageMeta(
        provider=provider,
        endpoint=endpoint,
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash=stable_hash(payload),
        source_authority=provider,
    )


def _resp(provider: str, endpoint: str, payload=None, *, error: ProviderError = None):
    return AdapterResponse(
        data=payload,
        lineage=_lineage(provider, endpoint, payload),
        error=error,
    )


def _snapshot(ticker: str, scan_id: str, day: date, market_cap: float = 100_000_000):
    asof = us_equity_session_close_timestamp(day)
    return UniverseSnapshot(
        universe_snapshot_id=f"{scan_id}-{ticker}",
        scan_id=scan_id,
        ticker=ticker,
        asof_timestamp=asof,
        price=10.0,
        market_cap=market_cap,
        primary_exchange="NASDAQ",
        security_type="common_stock",
        liquidity_score=1.0,
        hazard_score=10.0,
        operating_universe_inclusion=True,
        source_lineage_hash=f"lineage-{scan_id}-{ticker}",
    )


def _canonical_scan(db_session, day: date, tickers):
    scan_id = f"scan-{day.isoformat()}"
    scan = UniverseScan(
        scan_id=scan_id,
        trading_date=day.isoformat(),
        asof_timestamp=us_equity_session_close_timestamp(day),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        duplicate_symbol_count=0,
        run_status="finished",
    )
    db_session.add(scan)
    db_session.flush()
    db_session.add(CanonicalUniverseScan(trading_date=day.isoformat(), scan_id=scan_id))
    for ticker, market_cap in tickers.items():
        db_session.add(_snapshot(ticker, scan_id, day, market_cap))
    db_session.flush()
    return scan


class FakePolygonAdapter:
    def __init__(self, details):
        self.details = details
        self.calls = []

    def get_ticker_details(self, ticker, *, date_str=None, asof=None):
        self.calls.append((ticker.upper(), date_str))
        payload = self.details.get((ticker.upper(), date_str), self.details.get(ticker.upper()))
        if isinstance(payload, ProviderError):
            return _resp("polygon", "/v3/reference/tickers", None, error=payload)
        return _resp("polygon", "/v3/reference/tickers", payload)


class FakeFmpAdapter:
    def __init__(self, profiles=None, bars=None):
        self.profiles = profiles or {}
        self.bars = bars or {}

    def get_company_profile(self, ticker):
        return _resp("fmp", "/stable/profile", self.profiles.get(ticker.upper()))

    def get_historical_price(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        *,
        adjusted=False,
        require_split_adjusted_close=True,
        require_adjusted_close=False,
    ):
        return _resp("fmp", "/stable/historical-price-eod/dividend-adjusted", self.bars.get(ticker.upper(), []))


def _bar(day: date, adj_close: float):
    return SimpleNamespace(date=day.isoformat(), adj_close=adj_close, close=adj_close)


def test_m3_schema_has_server_defaults(db_session):
    inspector = inspect(db_session.bind)
    defaults = {
        table: {
            column["name"]: column.get("default")
            for column in inspector.get_columns(table)
        }
        for table in (
            "firm_sector_assignments_history",
            "firm_sector_assignments",
            "sector_returns_daily",
            "sector_change_log",
        )
    }

    assert defaults["firm_sector_assignments_history"]["source"] is not None
    assert defaults["firm_sector_assignments_history"]["created_at"] is not None
    assert defaults["firm_sector_assignments"]["source"] is not None
    assert defaults["firm_sector_assignments"]["created_at"] is not None
    assert defaults["firm_sector_assignments"]["updated_at"] is not None
    assert defaults["sector_returns_daily"]["source"] is not None
    assert defaults["sector_returns_daily"]["point_in_time_passed"] is not None
    assert defaults["sector_returns_daily"]["formation_cohort_passed"] is not None
    assert defaults["sector_returns_daily"]["created_at"] is not None
    assert defaults["sector_returns_daily"]["updated_at"] is not None
    assert defaults["sector_change_log"]["detected_at"] is not None


def test_sector_return_pit_booleans_default_false_on_raw_insert(db_session):
    db_session.execute(text(
        """
        INSERT INTO sector_returns_daily (
            date, sector, return_6mo, sector_rank, sector_rank_normalized,
            n_sectors, n_firms_in_sector, total_market_cap_in_sector,
            sic_to_sector_map_version, formation_date
        )
        VALUES (
            '2026-06-04', 'Information Technology', 0.1, 1, 0.5,
            1, 1, 100000000, :version, '2025-12-04'
        )
        """
    ), {"version": SIC_TO_SECTOR_MAP_VERSION})
    row = db_session.execute(text(
        """
        SELECT point_in_time_passed, formation_cohort_passed
        FROM sector_returns_daily
        WHERE date = '2026-06-04' AND sector = 'Information Technology'
        """
    )).one()

    assert bool(row.point_in_time_passed) is False
    assert bool(row.formation_cohort_passed) is False


def test_sic_to_sector_map_is_versioned_and_roughly_gics_sized():
    assert SIC_TO_SECTOR_MAP_VERSION == "POLYGON_SIC_PREFIX_V3_2026_06_05"
    assert len(SECTORS) == 11
    assert set(major_group_map().values()) <= set(SECTORS)
    assert all(sector_for_sic(f"{major:02d}00") in SECTORS for major in range(1, 100))
    assert sector_for_sic("6022") == "Financials"
    assert sector_for_sic("7372") == "Information Technology"
    assert sector_for_sic("2834") == "Health Care"
    assert sector_for_sic("2812") == "Materials"
    assert sector_for_sic("3531") == "Industrials"
    assert sector_for_sic("3571") == "Information Technology"
    assert sector_for_sic("1311") == "Energy"
    assert sector_for_sic("6500") == "Real Estate"
    assert sector_for_sic(None) is None
    assert canonical_sector_from_fmp("Financial Services") == "Financials"
    assert canonical_sector_from_fmp("Consumer Cyclical") == "Consumer Discretionary"


def test_sector_history_intervals_are_point_in_time(db_session):
    first = ResolvedSectorAssignment(
        ticker="BBBY",
        asof_date=date(2022, 6, 1),
        sector="Consumer Discretionary",
        source=SOURCE_POLYGON_SIC,
        sic_code="5700",
        diagnostics=[],
        lineage_ids=[],
        lineage_hashes=[],
    )
    second = ResolvedSectorAssignment(
        ticker="BBBY",
        asof_date=date(2026, 6, 1),
        sector="Financials",
        source=SOURCE_POLYGON_SIC,
        sic_code="6022",
        diagnostics=[],
        lineage_ids=[],
        lineage_hashes=[],
    )

    assert write_sector_assignment_interval(db_session, first)
    assert write_sector_assignment_interval(db_session, second)

    old = load_sector_assignments_at(db_session, tickers=["BBBY"], asof_date=date(2023, 1, 1))
    new = load_sector_assignments_at(db_session, tickers=["BBBY"], asof_date=date(2026, 6, 2))

    assert old["BBBY"].sector == "Consumer Discretionary"
    assert old["BBBY"].sic_code == "5700"
    assert new["BBBY"].sector == "Financials"
    assert new["BBBY"].sic_code == "6022"
    current = db_session.get(FirmSectorAssignment, "BBBY")
    assert current.sector == "Financials"


def test_sector_history_out_of_order_write_does_not_overlap(db_session):
    current = ResolvedSectorAssignment(
        ticker="BBBY",
        asof_date=date(2026, 6, 1),
        sector="Consumer Discretionary",
        source=SOURCE_POLYGON_SIC,
        sic_code="5961",
        diagnostics=[],
        lineage_ids=[],
        lineage_hashes=[],
    )
    older = ResolvedSectorAssignment(
        ticker="BBBY",
        asof_date=date(2022, 6, 1),
        sector="Consumer Discretionary",
        source=SOURCE_POLYGON_SIC,
        sic_code="5700",
        diagnostics=[],
        lineage_ids=[],
        lineage_hashes=[],
    )

    assert write_sector_assignment_interval(db_session, current)
    assert write_sector_assignment_interval(db_session, older)

    rows = (
        db_session.query(FirmSectorAssignmentHistory)
        .filter_by(ticker="BBBY")
        .order_by(FirmSectorAssignmentHistory.valid_from.asc())
        .all()
    )
    assert [(row.valid_from, row.valid_to) for row in rows] == [
        (date(2022, 6, 1), date(2026, 6, 1)),
        (date(2026, 6, 1), date(9999, 12, 31)),
    ]
    assert load_sector_assignments_at(
        db_session, tickers=["BBBY"], asof_date=date(2023, 1, 1)
    )["BBBY"].sic_code == "5700"
    assert load_sector_assignments_at(
        db_session, tickers=["BBBY"], asof_date=date(2026, 6, 2)
    )["BBBY"].sic_code == "5961"


def test_sector_history_direct_overlap_insert_blocked_on_postgres(db_session):
    if db_session.bind.dialect.name != "postgresql":
        pytest.skip("PostgreSQL exclusion constraint is verified on real PG")
    db_session.add(FirmSectorAssignmentHistory(
        ticker="OVLP",
        sector="Financials",
        source=SOURCE_POLYGON_SIC,
        sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
        valid_from=date(2024, 1, 1),
        valid_to=date(2026, 1, 1),
    ))
    db_session.flush()
    db_session.add(FirmSectorAssignmentHistory(
        ticker="OVLP",
        sector="Financials",
        source=SOURCE_POLYGON_SIC,
        sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
        valid_from=date(2025, 1, 1),
        valid_to=date(2027, 1, 1),
    ))

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_null_sic_uses_fmp_fallback_and_records_source():
    polygon = FakePolygonAdapter({
        ("FRC", "2022-06-01"): SimpleNamespace(
            ticker="FRC",
            sic_code=None,
            sic_description=None,
        )
    })
    fmp = FakeFmpAdapter(profiles={
        "FRC": SimpleNamespace(sector="Financial Services", industry="Banks")
    })

    resolved = resolve_sector_assignment(
        ticker="FRC",
        asof_date=date(2022, 6, 1),
        polygon_adapter=polygon,
        fmp_adapter=fmp,
        asof_timestamp=_ts(),
    )

    assert resolved.resolved
    assert resolved.source == SOURCE_FMP_FALLBACK
    assert resolved.sector == "Financials"
    assert "polygon_sic_null_or_unmapped" in resolved.diagnostics


def test_shumway_delisting_adjustment_when_end_price_missing():
    ret, applied = adjusted_return(
        [_bar(date(2023, 1, 3), 100.0), _bar(date(2023, 3, 10), 80.0)],
        start_date=date(2023, 1, 3),
        end_date=date(2023, 6, 30),
        delisted_date=date(2023, 3, 10),
    )

    assert applied is True
    assert ret == pytest.approx((0.8 * 0.7) - 1.0)


def test_acquisition_delisting_does_not_apply_shumway_penalty():
    ret, applied = adjusted_return(
        [_bar(date(2023, 1, 3), 100.0), _bar(date(2023, 3, 10), 140.0)],
        start_date=date(2023, 1, 3),
        end_date=date(2023, 6, 30),
        delisted_date=date(2023, 3, 10),
        delisting_reason="cash merger acquisition",
    )

    assert applied is False
    assert ret == pytest.approx(0.40)


def test_ambiguous_failed_acquisition_delisting_applies_shumway():
    ret, applied = adjusted_return(
        [_bar(date(2023, 1, 3), 100.0), _bar(date(2023, 3, 10), 80.0)],
        start_date=date(2023, 1, 3),
        end_date=date(2023, 6, 30),
        delisted_date=date(2023, 3, 10),
        delisting_reason="failed acquisition attempt, bankrupt",
    )

    assert applied is True
    assert ret == pytest.approx((0.8 * 0.7) - 1.0)


def test_polygon_404_delisting_probe_uses_last_available_bar(db_session):
    polygon = FakePolygonAdapter({
        ("SIVB", "2023-06-30"): ProviderError(
            provider="polygon",
            endpoint="/v3/reference/tickers/SIVB",
            status_code=404,
            error_type="http",
            message="Polygon HTTP 404",
            retryable=False,
        )
    })

    resolved = _resolve_delisted_dates_for_missing_end_prices(
        db_session,
        polygon,
        tickers=["SIVB"],
        bars_by_ticker={
            "SIVB": [
                _bar(date(2023, 1, 3), 100.0),
                _bar(date(2023, 3, 10), 80.0),
            ]
        },
        evidence_day=date(2023, 6, 30),
        asof=_ts(),
        job_run_id=None,
    )

    assert resolved == {"SIVB": date(2023, 3, 10)}


def test_sector_returns_use_formation_sector_not_current_membership():
    formation_day = date(2026, 1, 2)
    evidence_day = date(2026, 6, 4)
    snapshots = [
        SimpleNamespace(ticker="MOVE", market_cap=100.0),
        SimpleNamespace(ticker="STAY", market_cap=100.0),
    ]
    formation_assignments = {
        "MOVE": SectorAssignmentSnapshot(
            ticker="MOVE",
            sector="Energy",
            source=SOURCE_POLYGON_SIC,
            sic_code="1311",
            valid_from=date(2025, 1, 1),
            valid_to=date(2026, 3, 1),
        ),
        "STAY": SectorAssignmentSnapshot(
            ticker="STAY",
            sector="Information Technology",
            source=SOURCE_POLYGON_SIC,
            sic_code="7372",
            valid_from=date(2025, 1, 1),
            valid_to=date(9999, 12, 31),
        ),
    }
    bars = {
        "MOVE": [_bar(formation_day, 10.0), _bar(evidence_day, 12.0)],
        "STAY": [_bar(formation_day, 10.0), _bar(evidence_day, 9.0)],
    }

    components, diagnostics = build_sector_return_components(
        formation_snapshots=snapshots,
        assignments_by_ticker=formation_assignments,
        bars_by_ticker=bars,
        evidence_date=evidence_day,
        formation_date=formation_day,
    )
    returns = compute_sector_return_snapshots(
        components=components,
        asof_date=evidence_day,
        formation_date=formation_day,
    )
    by_sector = {row.sector: row for row in returns}

    assert diagnostics == []
    assert by_sector["Energy"].return_6mo == pytest.approx(0.20)
    assert by_sector["Energy"].sector_rank_normalized == pytest.approx(0.75)
    assert by_sector["Information Technology"].sector_rank_normalized == pytest.approx(0.25)


def test_sector_return_pit_gate_uses_formation_components_per_sector():
    formation_day = date(2026, 1, 2)
    evidence_day = date(2026, 6, 4)
    components = [
        SectorReturnComponent(
            ticker="COVERED",
            sector="Energy",
            market_cap=100.0,
            return_6mo=0.30,
            sector_history_coverage_years=3.2,
        ),
        SectorReturnComponent(
            ticker="ITGOOD",
            sector="Information Technology",
            market_cap=100.0,
            return_6mo=0.40,
            sector_history_coverage_years=3.5,
        ),
        SectorReturnComponent(
            ticker="ITBAD",
            sector="Information Technology",
            market_cap=100.0,
            return_6mo=0.50,
            sector_history_coverage_years=0.9,
        ),
    ]

    returns = compute_sector_return_snapshots(
        components=components,
        asof_date=evidence_day,
        formation_date=formation_day,
    )
    by_sector = {row.sector: row for row in returns}

    assert by_sector["Energy"].point_in_time_passed is True
    assert by_sector["Energy"].sector_history_coverage_years == pytest.approx(3.2)
    assert by_sector["Information Technology"].point_in_time_passed is False
    assert by_sector["Information Technology"].sector_history_coverage_years == pytest.approx(0.9)


def test_assembler_sets_polygon_sic_and_detector_rejects_without_pit_proof():
    snap = SimpleNamespace(
        ticker="FIRE",
        universe_snapshot_id="snap-fire",
        asof_timestamp=_ts(),
        source_lineage_hash="snap-hash",
        price=10.0,
        market_cap=100_000_000,
        primary_exchange="NASDAQ",
        security_type="common_stock",
        operating_universe_inclusion=True,
        liquidity_score=1.0,
        hazard_score=10.0,
    )
    assignment = SectorAssignmentSnapshot(
        ticker="FIRE",
        sector="Information Technology",
        source=SOURCE_POLYGON_SIC,
        sic_code="7372",
        valid_from=date(2023, 1, 1),
        valid_to=date(9999, 12, 31),
        sector_history_coverage_years=3.1,
    )
    sector_return = SectorReturnSnapshot(
        date=date(2026, 6, 4),
        sector="Information Technology",
        return_6mo=0.30,
        return_6mo_ew=0.25,
        return_1mo=0.02,
        return_3mo=0.10,
        sector_rank=3,
        sector_rank_normalized=0.833333,
        n_sectors=3,
        n_firms_in_sector=8,
        total_market_cap_in_sector=500_000_000,
        formation_date=date(2025, 12, 4),
        point_in_time_passed=False,
        formation_cohort_passed=False,
        sector_history_coverage_years=3.1,
    )

    assembled = assemble_m3_daily(
        snapshots=[snap],
        assignments_by_ticker={"FIRE": assignment},
        sector_returns_by_sector={"Information Technology": sector_return},
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=_ts(),
        decision_date="2026-06-04",
        evidence_session_date="2026-06-04",
        next_execution_session="2026-06-05",
        allow_undercoverage=True,
    )
    inp = assembled.inputs[0]
    assert inp.market_data["sector_taxonomy_source"] == "POLYGON_SIC"

    result = M3Detector().detect(inp)
    assert not result.has_signal
    assert result.features.features["rejection_reason"] == "sector_return_not_point_in_time"
    assert result.features.point_in_time_passed is False


def test_assembler_routes_undercovered_sector_returns_to_shadow_only():
    snap = SimpleNamespace(
        ticker="FIRE",
        universe_snapshot_id="snap-fire",
        asof_timestamp=_ts(),
        source_lineage_hash="snap-hash",
        price=10.0,
        market_cap=100_000_000,
        primary_exchange="NASDAQ",
        security_type="common_stock",
        operating_universe_inclusion=True,
        liquidity_score=1.0,
        hazard_score=10.0,
    )
    assignment = SectorAssignmentSnapshot(
        ticker="FIRE",
        sector="Information Technology",
        source=SOURCE_POLYGON_SIC,
        sic_code="7372",
        valid_from=date(2024, 6, 1),
        valid_to=date(9999, 12, 31),
        sector_history_coverage_years=2.0,
        last_verified=date(2026, 6, 4),
    )
    sector_return = SectorReturnSnapshot(
        date=date(2026, 6, 4),
        sector="Information Technology",
        return_6mo=0.30,
        return_6mo_ew=0.25,
        return_1mo=0.02,
        return_3mo=0.10,
        sector_rank=3,
        sector_rank_normalized=0.833333,
        n_sectors=3,
        n_firms_in_sector=8,
        total_market_cap_in_sector=500_000_000,
        formation_date=date(2025, 12, 4),
        point_in_time_passed=False,
        formation_cohort_passed=True,
        sector_history_coverage_years=2.0,
    )

    production = assemble_m3_daily(
        snapshots=[snap],
        assignments_by_ticker={"FIRE": assignment},
        sector_returns_by_sector={"Information Technology": sector_return},
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=_ts(),
        decision_date="2026-06-04",
        evidence_session_date="2026-06-04",
        next_execution_session="2026-06-05",
    )
    shadow = assemble_m3_daily(
        snapshots=[snap],
        assignments_by_ticker={"FIRE": assignment},
        sector_returns_by_sector={"Information Technology": sector_return},
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=_ts(),
        decision_date="2026-06-04",
        evidence_session_date="2026-06-04",
        next_execution_session="2026-06-05",
        pattern_id=SHADOW_PATTERN_ID,
        allow_undercoverage=True,
    )

    assert production.inputs == []
    assert production.diagnostics[0].diagnostic_type == "sector_history_coverage_below_minimum"
    assert len(shadow.inputs) == 1
    assert shadow.inputs[0].market_data["sector_return_point_in_time_passed"] is False
    assert shadow.inputs[0].market_data["field_confidence"]["current_sector_assignment_coverage"] < 1


def test_m3_daily_job_persists_pit_sector_returns_and_signals(db_session):
    evidence_day = date(2026, 6, 4)
    formation_day = nth_previous_session(evidence_day, 126)
    tickers = {"FIRE": 100_000_000, "MID": 90_000_000, "LOW": 80_000_000}
    _canonical_scan(db_session, evidence_day, tickers)
    _canonical_scan(db_session, formation_day, tickers)
    for ticker, sector, sic in (
        ("FIRE", "Information Technology", "7372"),
        ("MID", "Energy", "1311"),
        ("LOW", "Health Care", "2834"),
    ):
        db_session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=sector,
            sic_code=sic,
            source=SOURCE_POLYGON_SIC,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=date(2022, 1, 1),
            valid_to=date(9999, 12, 31),
        ))
        db_session.add(FirmSectorAssignment(
            ticker=ticker,
            sector=sector,
            source=SOURCE_POLYGON_SIC,
            classification_date=evidence_day,
            last_verified=evidence_day,
        ))
    bars = {
        "FIRE": [_bar(formation_day, 10.0), _bar(evidence_day, 13.0)],
        "MID": [_bar(formation_day, 10.0), _bar(evidence_day, 11.0)],
        "LOW": [_bar(formation_day, 10.0), _bar(evidence_day, 9.0)],
    }
    job = M3DailyAssemblyJob(
        db_session,
        polygon_adapter=FakePolygonAdapter({}),
        fmp_adapter=FakeFmpAdapter(bars=bars),
        run_timestamp=_ts(),
        refresh_sector_history=False,
    )

    result = run_job(db_session, job, params={"run_timestamp": _ts().isoformat()})

    assert result.status == "finished"
    assert result.metrics["formation_date"] == formation_day.isoformat()
    assert result.metrics["sector_return_count"] == 3
    assert db_session.query(SectorReturnDaily).count() == 3
    signal = db_session.query(SignalRegistry).filter_by(pattern_id="M3", ticker="FIRE").one()
    assert signal.signal_horizon == "15d"
    assert signal.signal_timestamp.isoformat().startswith("2026-06-04T20:00:00")
    assert result.metrics["orchestration"]["total_signals_persisted"] == 1


def test_m3_daily_job_undercoverage_persists_shadow_only(db_session):
    evidence_day = date(2026, 6, 4)
    formation_day = nth_previous_session(evidence_day, 126)
    tickers = {"FIRE": 100_000_000, "MID": 90_000_000, "LOW": 80_000_000}
    _canonical_scan(db_session, evidence_day, tickers)
    _canonical_scan(db_session, formation_day, tickers)
    for ticker, sector, sic in (
        ("FIRE", "Information Technology", "7372"),
        ("MID", "Energy", "1311"),
        ("LOW", "Health Care", "2834"),
    ):
        db_session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=sector,
            sic_code=sic,
            source=SOURCE_POLYGON_SIC,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=date(2025, 1, 1),
            valid_to=date(9999, 12, 31),
        ))
        db_session.add(FirmSectorAssignment(
            ticker=ticker,
            sector=sector,
            source=SOURCE_POLYGON_SIC,
            classification_date=evidence_day,
            last_verified=evidence_day,
        ))
    bars = {
        "FIRE": [_bar(formation_day, 10.0), _bar(evidence_day, 13.0)],
        "MID": [_bar(formation_day, 10.0), _bar(evidence_day, 11.0)],
        "LOW": [_bar(formation_day, 10.0), _bar(evidence_day, 9.0)],
    }
    job = M3DailyAssemblyJob(
        db_session,
        polygon_adapter=FakePolygonAdapter({}),
        fmp_adapter=FakeFmpAdapter(bars=bars),
        run_timestamp=_ts(),
        refresh_sector_history=False,
    )

    result = run_job(db_session, job, params={"run_timestamp": _ts().isoformat()})

    assert result.status == "finished"
    assert result.metrics["assembly"]["assembled_count"] == 0
    assert result.metrics["shadow_assembly"]["assembled_count"] == 3
    assert db_session.query(SignalRegistry).filter_by(pattern_id="M3").count() == 0
    shadow = db_session.query(SignalRegistry).filter_by(pattern_id=SHADOW_PATTERN_ID, ticker="FIRE").one()
    assert shadow.signal_status == "shadow"
    assert shadow.point_in_time_passed is False


def test_m3_daily_job_mixed_formation_undercoverage_shadows_only_that_sector(db_session):
    evidence_day = date(2026, 6, 4)
    formation_day = nth_previous_session(evidence_day, 126)
    current_tickers = {
        "FIRE": 100_000_000,
        "ENGY": 90_000_000,
        "LOW": 80_000_000,
        "CONS": 70_000_000,
        "FIN": 60_000_000,
    }
    formation_tickers = dict(current_tickers)
    formation_tickers["ITBAD"] = 100_000_000
    _canonical_scan(db_session, evidence_day, current_tickers)
    _canonical_scan(db_session, formation_day, formation_tickers)
    for ticker, sector, sic, valid_from in (
        ("FIRE", "Information Technology", "7372", date(2022, 1, 1)),
        ("ITBAD", "Information Technology", "7372", date(2025, 3, 1)),
        ("ENGY", "Energy", "1311", date(2022, 1, 1)),
        ("LOW", "Health Care", "2834", date(2022, 1, 1)),
        ("CONS", "Consumer Staples", "2000", date(2022, 1, 1)),
        ("FIN", "Financials", "6022", date(2022, 1, 1)),
    ):
        db_session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=sector,
            sic_code=sic,
            source=SOURCE_POLYGON_SIC,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=valid_from,
            valid_to=date(9999, 12, 31),
        ))
        if ticker in current_tickers:
            db_session.add(FirmSectorAssignment(
                ticker=ticker,
                sector=sector,
                source=SOURCE_POLYGON_SIC,
                classification_date=evidence_day,
                last_verified=evidence_day,
            ))
    bars = {
        "FIRE": [_bar(formation_day, 10.0), _bar(evidence_day, 15.0)],
        "ITBAD": [_bar(formation_day, 10.0), _bar(evidence_day, 16.0)],
        "ENGY": [_bar(formation_day, 10.0), _bar(evidence_day, 15.0)],
        "LOW": [_bar(formation_day, 10.0), _bar(evidence_day, 11.0)],
        "CONS": [_bar(formation_day, 10.0), _bar(evidence_day, 10.0)],
        "FIN": [_bar(formation_day, 10.0), _bar(evidence_day, 9.5)],
    }
    job = M3DailyAssemblyJob(
        db_session,
        polygon_adapter=FakePolygonAdapter({}),
        fmp_adapter=FakeFmpAdapter(bars=bars),
        run_timestamp=_ts(),
        refresh_sector_history=False,
    )

    result = run_job(db_session, job, params={"run_timestamp": _ts().isoformat()})

    assert result.status == "finished"
    it_return = db_session.get(
        SectorReturnDaily,
        {"date": evidence_day, "sector": "Information Technology"},
    )
    energy_return = db_session.get(
        SectorReturnDaily,
        {"date": evidence_day, "sector": "Energy"},
    )
    assert it_return.n_firms_in_sector == 2
    assert it_return.point_in_time_passed is False
    assert it_return.sector_history_coverage_years < 3.0
    assert energy_return.point_in_time_passed is True
    assert db_session.query(SignalRegistry).filter_by(pattern_id="M3", ticker="FIRE").count() == 0
    assert db_session.query(SignalRegistry).filter_by(pattern_id=SHADOW_PATTERN_ID, ticker="FIRE").count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="M3", ticker="ENGY").count() == 1


def test_m3_current_assignment_staleness_reduces_signal_confidence(db_session):
    evidence_day = date(2026, 6, 4)
    formation_day = nth_previous_session(evidence_day, 126)
    tickers = {"FIRE": 100_000_000, "MID": 90_000_000, "LOW": 80_000_000}
    _canonical_scan(db_session, evidence_day, tickers)
    _canonical_scan(db_session, formation_day, tickers)
    for ticker, sector, sic in (
        ("FIRE", "Information Technology", "7372"),
        ("MID", "Energy", "1311"),
        ("LOW", "Health Care", "2834"),
    ):
        db_session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=sector,
            sic_code=sic,
            source=SOURCE_POLYGON_SIC,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=date(2022, 1, 1),
            valid_to=date(9999, 12, 31),
        ))
        db_session.add(FirmSectorAssignment(
            ticker=ticker,
            sector=sector,
            source=SOURCE_POLYGON_SIC,
            classification_date=date(2026, 1, 1),
            last_verified=date(2026, 1, 1),
        ))
    bars = {
        "FIRE": [_bar(formation_day, 10.0), _bar(evidence_day, 13.0)],
        "MID": [_bar(formation_day, 10.0), _bar(evidence_day, 11.0)],
        "LOW": [_bar(formation_day, 10.0), _bar(evidence_day, 9.0)],
    }
    job = M3DailyAssemblyJob(
        db_session,
        polygon_adapter=FakePolygonAdapter({}),
        fmp_adapter=FakeFmpAdapter(bars=bars),
        run_timestamp=_ts(),
        refresh_sector_history=False,
    )

    result = run_job(db_session, job, params={"run_timestamp": _ts().isoformat()})

    assert result.status == "finished"
    signal = db_session.query(SignalRegistry).filter_by(pattern_id="M3", ticker="FIRE").one()
    assert signal.data_confidence < 1.0
