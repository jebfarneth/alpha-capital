"""
Feature assembly tests.

Covers all acceptance criteria from Engineering/FeatureAssembly.md:
  - Registry represents all 17 pattern ids with correct statuses.
  - M4 daily assembler produces PatternInput objects with lineage from fixtures.
  - Orchestration can produce feature_snapshots > 0 from M4 assembled inputs.
  - Future-bar leak test fails closed.
  - Missing/insufficient-history test proves no zero-fill.
  - Existing orchestration behavior remains compatible.
"""

from __future__ import annotations

import json
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Dict, List
from zoneinfo import ZoneInfo

import pytest

from alpha.assembly.framework import (
    AssembledField,
    AssemblyDiagnostic,
    FieldPresence,
    PatternAssemblyResult,
    build_pattern_input,
    validate_assembled_fields,
)
from alpha.assembly.m4_daily import (
    DailyBar,
    assemble_m4_daily,
)
from alpha.assembly.registry import AssemblerStatus, AssemblyRegistry
from alpha.data.contracts import stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    FeatureSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.runner import run_job
from alpha.patterns.contracts import (
    BasePatternDetector,
    FidelityTier,
    PatternDetectionResult,
    PatternFeatures,
    PatternId,
    PatternInput,
    PatternSignal,
    PatternTrack,
    RouteClass,
    SignalDirection,
    ThesisCategory,
)
from alpha.patterns.m4 import M4Detector
from alpha.evidence.writer import record_data_lineage
import alpha.market_calendar as market_calendar
from alpha.market_calendar import (
    is_us_equity_session,
    nyse_early_closes,
    nyse_holidays,
    resolve_us_equity_session,
    us_equity_session_close_time,
    us_equity_session_close_timestamp,
)


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


def _make_bars(
    n: int = 252,
    base_high: float = 9.0,
    final_high: float = 10.0,
    evidence_date: str = "2026-05-20",
    evidence_split_adjusted_close: float | None = None,
    evidence_close: float | None = None,
    source_ts: datetime | None = None,
) -> List[DailyBar]:
    """Generate n prior-session bars plus one evidence-session bar."""
    evidence_day = date_type.fromisoformat(evidence_date)
    prior_days = _prior_weekdays(evidence_day, n)
    bars: List[DailyBar] = []
    for idx, day in enumerate(prior_days):
        high = base_high if idx < n - 1 else final_high
        bars.append(DailyBar(
            date=day.isoformat(),
            open=high - 0.5,
            high=high,
            low=high - 1.0,
            close=high - 0.2,
            volume=100_000,
            split_adjusted_close=high,
            adj_close=high,
            source_timestamp=source_ts or _ts(),
            source_provider="FMP",
            lineage_hash="bar-lineage-hash",
        ))
    evidence_value = (
        evidence_split_adjusted_close if evidence_split_adjusted_close is not None else final_high
    )
    evidence_raw_close = (
        evidence_close if evidence_close is not None else evidence_value
    )
    bars.append(DailyBar(
        date=evidence_date,
        open=evidence_raw_close - 0.5,
        high=max(evidence_raw_close, evidence_value),
        low=evidence_raw_close - 1.0,
        close=evidence_raw_close,
        volume=200_000,
        split_adjusted_close=evidence_value,
        adj_close=evidence_value,
        source_timestamp=source_ts or _ts(),
        source_provider="FMP",
        lineage_hash="bar-lineage-hash",
    ))
    return bars


def _make_breakout_bars(
    n: int = 252,
    high_52w: float = 10.0,
    close_price: float = 10.5,
    evidence_date: str = "2026-05-20",
    source_ts: datetime | None = None,
) -> List[DailyBar]:
    """Generate bars where evidence-session split-adjusted close is the M4 price."""
    return _make_bars(
        n=n,
        base_high=high_52w - 1.0,
        final_high=high_52w,
        evidence_date=evidence_date,
        evidence_split_adjusted_close=close_price,
        evidence_close=close_price,
        source_ts=source_ts,
    )


def _prior_weekdays(end_day: date_type, n: int) -> List[date_type]:
    days: List[date_type] = []
    cursor = end_day - timedelta(days=1)
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def _day_after_thanksgiving(year: int) -> date_type:
    cursor = date_type(year, 11, 1)
    thursdays = 0
    while True:
        if cursor.weekday() == 3:
            thursdays += 1
            if thursdays == 4:
                return cursor + timedelta(days=1)
        cursor += timedelta(days=1)


def _make_snapshot(
    ticker: str = "ACME",
    price: float = 10.5,
    universe_snapshot_id: str | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "universe_snapshot_id": universe_snapshot_id or f"snap-{ticker}",
        "asof_timestamp": _ts(),
        "price": price,
        "market_cap": 75_000_000,
        "primary_exchange": "NASDAQ",
        "security_type": "common_stock",
        "operating_universe_inclusion": True,
        "source_lineage_hash": "snap-lineage-hash",
    }


def _setup_canonical_universe(
    db_session, trading_date="2026-05-20", tickers=None, prices=None,
):
    """Create canonical universe with included snapshots for test DB."""
    if tickers is None:
        tickers = ["ACME", "BETA"]
    if prices is None:
        prices = {t: 10.5 for t in tickers}

    scan = UniverseScan(
        scan_id="test-scan",
        trading_date=trading_date,
        asof_timestamp=_ts(),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="screener-hash",
    )
    db_session.add(scan)
    db_session.flush()

    db_session.add(CanonicalUniverseScan(
        trading_date=trading_date,
        scan_id="test-scan",
        selection_reason="test",
    ))
    db_session.flush()

    snapshots = []
    for ticker in tickers:
        snap = UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id="test-scan",
            ticker=ticker,
            asof_timestamp=_ts(),
            market_cap=75_000_000,
            price=prices.get(ticker, 10.5),
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash="snap-lineage-hash",
        )
        db_session.add(snap)
        snapshots.append(snap)

    db_session.flush()
    return scan, snapshots


# ===================================================================
# 1. Registry: all 17 pattern ids with correct statuses
# ===================================================================


class TestAssemblyRegistry:
    def test_all_17_patterns_represented(self):
        registry = AssemblyRegistry()
        entries = registry.all_entries()
        assert len(entries) == 17
        assert {e.pattern_id for e in entries} == set(PatternId.ALL)

    def test_default_statuses_correct(self):
        registry = AssemblyRegistry()
        # No assemblers registered: all detectors are detector_only, rest reserved
        detector_patterns = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "I1", "I8"}
        reserved_patterns = set(PatternId.ALL) - detector_patterns

        for pid in detector_patterns:
            assert registry.status(pid) == AssemblerStatus.DETECTOR_ONLY, pid
        for pid in reserved_patterns:
            assert registry.status(pid) == AssemblerStatus.RESERVED, pid

    def test_m4_implemented_when_assembler_registered(self):
        registry = AssemblyRegistry(assemblers={"M4": assemble_m4_daily})
        assert registry.status("M4") == AssemblerStatus.IMPLEMENTED
        assert registry.get("M4").assembler is assemble_m4_daily

    def test_disabled_overrides_everything(self):
        registry = AssemblyRegistry(
            assemblers={"M4": assemble_m4_daily},
            disabled={"M4"},
        )
        assert registry.status("M4") == AssemblerStatus.DISABLED
        assert registry.get("M4").assembler is None

    def test_implemented_ids_returns_only_implemented(self):
        registry = AssemblyRegistry(assemblers={"M4": assemble_m4_daily})
        assert registry.implemented_ids() == ["M4"]

    def test_diagnostics_returns_full_status_map(self):
        registry = AssemblyRegistry(assemblers={"M4": assemble_m4_daily})
        diag = registry.diagnostics()
        assert len(diag) == 17
        assert diag["M4"] == AssemblerStatus.IMPLEMENTED
        assert diag["M1"] == AssemblerStatus.DETECTOR_ONLY
        assert diag["I2"] == AssemblerStatus.RESERVED

    def test_unknown_pattern_id_raises(self):
        registry = AssemblyRegistry()
        with pytest.raises(KeyError, match="unknown"):
            registry.get("FAKE")


# ===================================================================
# 2. Framework: field-level lookahead enforcement
# ===================================================================


class TestFrameworkLookahead:
    def test_present_field_before_cutoff_passes(self):
        cutoff = datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc)
        fields = [
            AssembledField(
                name="price", value=10.5,
                presence=FieldPresence.PRESENT,
                source_timestamp=datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc),
                allowed_cutoff=cutoff,
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert validated == {"price": 10.5}
        assert rejected == []

    def test_present_field_after_cutoff_rejected(self):
        cutoff = datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc)
        fields = [
            AssembledField(
                name="price", value=10.5,
                presence=FieldPresence.PRESENT,
                source_timestamp=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
                allowed_cutoff=cutoff,
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert "price" not in validated
        assert len(rejected) == 1
        assert rejected[0].presence == FieldPresence.REJECTED_LOOKAHEAD
        assert "after cutoff" in rejected[0].rejection_reason

    def test_field_specific_allowed_cutoff_overrides_global_cutoff(self):
        bar_cutoff = datetime(2026, 5, 22, 20, 0, tzinfo=timezone.utc)
        universe_cutoff = datetime(2026, 5, 26, 8, 30, tzinfo=timezone.utc)
        fields = [
            AssembledField(
                name="operating_universe_inclusion",
                value=True,
                presence=FieldPresence.PRESENT,
                source_timestamp=universe_cutoff,
                allowed_cutoff=universe_cutoff,
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, bar_cutoff)
        assert validated == {"operating_universe_inclusion": True}
        assert rejected == []

    def test_missing_field_goes_to_rejected(self):
        cutoff = _ts()
        fields = [
            AssembledField(
                name="high_52w", value=None,
                presence=FieldPresence.MISSING,
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert "high_52w" not in validated
        assert len(rejected) == 1

    def test_unavailable_field_goes_to_rejected(self):
        cutoff = _ts()
        fields = [
            AssembledField(
                name="high_52w", value=None,
                presence=FieldPresence.UNAVAILABLE,
                rejection_reason="no_daily_bars_available",
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert "high_52w" not in validated
        assert len(rejected) == 1

    def test_present_field_without_timestamp_passes(self):
        """Fields without source_timestamp (e.g. trading_date) pass through."""
        cutoff = _ts()
        fields = [
            AssembledField(
                name="trading_date", value="2026-05-20",
                presence=FieldPresence.PRESENT,
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert validated == {"trading_date": "2026-05-20"}
        assert rejected == []

    def test_zero_value_preserved_not_coerced(self):
        """0 means observed zero — must not be treated as missing."""
        cutoff = _ts()
        fields = [
            AssembledField(
                name="volume", value=0,
                presence=FieldPresence.PRESENT,
                source_timestamp=_ts() - timedelta(hours=1),
            ),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff)
        assert validated["volume"] == 0


# ===================================================================
# 3. Market calendar/session resolver
# ===================================================================


class TestMarketSessionResolver:
    def test_early_close_timestamp_uses_1pm_et(self):
        assert us_equity_session_close_timestamp(
            date_type(2026, 11, 27)
        ) == datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)
        assert us_equity_session_close_timestamp(
            date_type(2026, 12, 24)
        ) == datetime(2026, 12, 24, 18, 0, tzinfo=timezone.utc)
        assert us_equity_session_close_timestamp(
            date_type(2026, 12, 23)
        ) == datetime(2026, 12, 23, 21, 0, tzinfo=timezone.utc)

    def test_early_close_summer_timestamp_is_dst_correct(self):
        assert us_equity_session_close_timestamp(
            date_type(2028, 7, 3)
        ) == datetime(2028, 7, 3, 17, 0, tzinfo=timezone.utc)

    def test_early_close_before_close_uses_prior_completed_session(self):
        run_ts = datetime(
            2026, 11, 27, 12, 59, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.evidence_session_date == "2026-11-25"
        assert resolved.next_execution_session == "2026-11-27"

    def test_early_close_at_close_uses_same_completed_session(self):
        run_ts = datetime(
            2026, 11, 27, 13, 0, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.evidence_session_date == "2026-11-27"
        assert resolved.next_execution_session == "2026-11-30"

    def test_early_close_after_close_uses_same_completed_session(self):
        run_ts = datetime(
            2026, 11, 27, 15, 0, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.evidence_session_date == "2026-11-27"
        assert resolved.next_execution_session == "2026-11-30"

    def test_non_early_close_regular_day_still_uses_4pm(self):
        before_close = resolve_us_equity_session(datetime(
            2026, 11, 25, 15, 59, tzinfo=ZoneInfo("America/New_York")
        ))
        at_close = resolve_us_equity_session(datetime(
            2026, 11, 25, 16, 0, tzinfo=ZoneInfo("America/New_York")
        ))

        assert before_close.evidence_session_date == "2026-11-24"
        assert at_close.evidence_session_date == "2026-11-25"

    def test_early_close_set_integrity_and_future_degradation(self):
        assert nyse_early_closes(2026) == {
            date_type(2026, 11, 27),
            date_type(2026, 12, 24),
        }
        assert nyse_early_closes(2027) == {date_type(2027, 11, 26)}
        assert nyse_early_closes(2028) == {
            date_type(2028, 7, 3),
            date_type(2028, 11, 24),
        }
        assert nyse_early_closes(2099) == set()
        for year in (2026, 2027, 2028):
            holidays = nyse_holidays(year)
            for day in nyse_early_closes(year):
                assert is_us_equity_session(day)
                assert day not in holidays

    def test_early_close_table_extension_alarm(self):
        table_year = max(market_calendar._NYSE_EARLY_CLOSES)
        next_black_friday = _day_after_thanksgiving(table_year + 1)
        extension_deadline = date_type(table_year, 1, 1)

        assert date_type.today() < extension_deadline, (
            "_NYSE_EARLY_CLOSES must be extended before the audited horizon "
            f"year begins; next recurring half-day is {next_black_friday}"
        )

    @pytest.mark.parametrize(
        ("year", "bad_day"),
        [
            (2026, date_type(2026, 12, 25)),
            (2026, date_type(2026, 11, 28)),
            (2026, date_type(2027, 11, 26)),
        ],
    )
    def test_early_close_integrity_guard_rejects_bad_entries(
        self,
        monkeypatch,
        year,
        bad_day,
    ):
        monkeypatch.setattr(
            market_calendar,
            "_NYSE_EARLY_CLOSES",
            {year: frozenset({bad_day})},
        )

        with pytest.raises(ValueError, match="invalid NYSE early-close date"):
            nyse_early_closes(year)

    def test_session_close_time_rejects_non_session(self):
        with pytest.raises(ValueError, match="not a regular U.S. equity session"):
            us_equity_session_close_time(date_type(2026, 12, 25))

    def test_memorial_day_2026_tuesday_premarket(self):
        run_ts = datetime(
            2026, 5, 26, 4, 0, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.decision_date == "2026-05-26"
        assert resolved.evidence_session_date == "2026-05-22"
        assert resolved.next_execution_session == "2026-05-26"
        assert resolved.is_premarket_decision_window is True
        assert is_us_equity_session(date_type(2026, 5, 25)) is False

    def test_weekday_after_close_uses_same_completed_session(self):
        run_ts = datetime(
            2026, 5, 20, 17, 0, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.decision_date == "2026-05-20"
        assert resolved.evidence_session_date == "2026-05-20"
        assert resolved.next_execution_session == "2026-05-21"
        assert resolved.is_premarket_decision_window is False

    def test_weekday_before_open_uses_prior_completed_session(self):
        run_ts = datetime(
            2026, 5, 20, 4, 0, tzinfo=ZoneInfo("America/New_York")
        )
        resolved = resolve_us_equity_session(run_ts)

        assert resolved.decision_date == "2026-05-20"
        assert resolved.evidence_session_date == "2026-05-19"
        assert resolved.next_execution_session == "2026-05-20"
        assert resolved.is_premarket_decision_window is True


# ===================================================================
# 4. M4 daily assembler: fixture-based assembly
# ===================================================================


class TestM4DailyAssembler:
    def test_basic_assembly_produces_pattern_inputs(self):
        snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": _make_breakout_bars(n=252, high_52w=10.0, close_price=10.5)}
        result = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars=bars,
            trading_date="2026-05-20",
            cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert len(result.inputs) == 1

        inp = result.inputs[0]
        assert inp.ticker == "ACME"
        assert inp.market_data["price"] == 10.5
        assert inp.market_data["high_52w"] is not None
        assert inp.market_data["n_sessions_in_window"] == 252
        assert inp.market_data["operating_universe_inclusion"] is True
        assert inp.market_data["decision_date"] == "2026-05-20"
        assert inp.market_data["evidence_session_date"] == "2026-05-20"
        assert inp.market_data["trading_date"] == "2026-05-20"
        assert inp.market_data["price_source"] == "evidence_session_split_adjusted_close"
        assert inp.universe_snapshot_id == "snap-ACME"
        assert len(inp.lineage_hashes) > 0

    def test_deterministic_for_fixed_fixtures(self):
        snapshots = [_make_snapshot("ACME")]
        bars = {"ACME": _make_breakout_bars(n=100)}
        r1 = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        r2 = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert len(r1.inputs) == len(r2.inputs)
        assert r1.inputs[0].market_data == r2.inputs[0].market_data
        assert r1.inputs[0].lineage_hashes == r2.inputs[0].lineage_hashes

    def test_multiple_tickers_assembled(self):
        snapshots = [
            _make_snapshot("ACME", price=10.5),
            _make_snapshot("BETA", price=8.0),
        ]
        bars = {
            "ACME": _make_breakout_bars(n=252, high_52w=10.0, close_price=10.5),
            "BETA": _make_breakout_bars(n=252, high_52w=7.5, close_price=8.0),
        }
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 2
        tickers = {inp.ticker for inp in result.inputs}
        assert tickers == {"ACME", "BETA"}

    def test_lineage_hashes_populated(self):
        snapshots = [_make_snapshot("ACME")]
        bars = {"ACME": _make_breakout_bars(n=100)}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        inp = result.inputs[0]
        assert "snap-lineage-hash" in inp.lineage_hashes
        assert "bar-lineage-hash" in inp.lineage_hashes

    def test_high_52w_date_populated_when_computable(self):
        snapshots = [_make_snapshot("ACME")]
        bars = {"ACME": _make_breakout_bars(n=100)}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        inp = result.inputs[0]
        assert "high_52w_date" in inp.market_data
        assert inp.market_data["high_52w_date"] is not None

    def test_security_type_and_exchange_included(self):
        snapshots = [_make_snapshot("ACME")]
        bars = {"ACME": _make_breakout_bars(n=100)}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        inp = result.inputs[0]
        assert inp.market_data["security_type"] == "common_stock"
        assert inp.market_data["primary_exchange"] == "NASDAQ"

    def test_high_52w_uses_split_adjusted_close_not_raw_high(self):
        snapshots = [_make_snapshot("ACME", price=13.0)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-18",
                open=98.0,
                high=100.0,
                low=95.0,
                close=100.0,
                volume=100_000,
                split_adjusted_close=10.0,
                adj_close=80.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-19",
                open=11.0,
                high=12.5,
                low=10.5,
                close=12.0,
                volume=100_000,
                split_adjusted_close=12.0,
                adj_close=70.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-20",
                open=13.0,
                high=13.5,
                low=12.5,
                close=13.0,
                volume=100_000,
                split_adjusted_close=13.0,
                adj_close=60.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.inputs[0].market_data["price"] == 13.0
        assert result.inputs[0].market_data["high_52w"] == 12.0
        assert result.inputs[0].market_data["high_52w_basis"] == (
            "split_adjusted_close_prior_252_sessions"
        )

    def test_m4_price_uses_evidence_split_adjusted_close_not_snapshot_or_raw_close(self):
        snapshots = [_make_snapshot("ACME", price=99.0)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-19",
                open=9.0,
                high=10.0,
                low=8.5,
                close=10.0,
                volume=100_000,
                split_adjusted_close=10.0,
                adj_close=10.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-20",
                open=20.0,
                high=21.0,
                low=19.0,
                close=20.0,
                volume=100_000,
                split_adjusted_close=11.0,
                adj_close=18.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.inputs[0].market_data["price"] == 11.0
        assert result.inputs[0].market_data["snapshot_price"] == 99.0
        assert result.inputs[0].market_data["evidence_close"] == 20.0
        assert result.inputs[0].market_data["price_source"] == (
            "evidence_session_split_adjusted_close"
        )

    def test_trading_date_bar_excluded_from_high_52w(self):
        snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-19",
                open=9.0,
                high=10.0,
                low=8.5,
                close=10.0,
                volume=100_000,
                split_adjusted_close=10.0,
                adj_close=10.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-20",
                open=50.0,
                high=99.0,
                low=49.0,
                close=99.0,
                volume=100_000,
                split_adjusted_close=99.0,
                adj_close=99.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.inputs[0].market_data["high_52w"] == 10.0
        assert result.inputs[0].market_data["high_52w_date"] == "2026-05-19"

    def test_memorial_day_decision_uses_friday_evidence_session(self):
        snapshots = [_make_snapshot("ACME", price=99.0)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-21",
                open=9.5,
                high=10.0,
                low=9.0,
                close=10.0,
                volume=100_000,
                split_adjusted_close=10.0,
                adj_close=10.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-22",
                open=11.0,
                high=11.5,
                low=10.5,
                close=11.0,
                volume=100_000,
                split_adjusted_close=11.0,
                adj_close=11.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}

        result = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars=bars,
            decision_date="2026-05-26",
            evidence_session_date="2026-05-22",
            cutoff_timestamp=_ts(),
        )

        assert result.assembled_count == 1
        inp = result.inputs[0]
        assert inp.market_data["decision_date"] == "2026-05-26"
        assert inp.market_data["evidence_session_date"] == "2026-05-22"
        assert inp.market_data["price"] == 11.0
        assert inp.market_data["high_52w"] == 10.0
        assert inp.market_data["high_52w_date"] == "2026-05-21"
        assert inp.market_data["lookback_end"] == "2026-05-21"

    def test_missing_split_adjusted_close_rejects_unadjusted_high(self):
        snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-19",
                open=9.0,
                high=100.0,
                low=8.5,
                close=100.0,
                volume=100_000,
                adj_close=90.0,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
            DailyBar(
                date="2026-05-20",
                open=10.0,
                high=10.5,
                low=9.8,
                close=10.4,
                volume=100_000,
                split_adjusted_close=10.4,
                adj_close=10.4,
                source_timestamp=_ts(),
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 0
        assert result.rejected_count == 1
        assert any(
            d.diagnostic_type == "split_adjusted_close_unavailable"
            for d in result.diagnostics
        )


# ===================================================================
# 4. Future-bar leak test: fails closed
# ===================================================================


class TestFutureBarLeak:
    def test_future_bar_rejected_produces_no_input(self):
        """A daily bar with source_timestamp after the cutoff must be rejected.

        This is the antagonistic M4 future-bar test required by the spec.
        The framework must reject future-contaminated fields and produce
        no detector input / feature snapshot from the contaminated data.
        """
        cutoff = datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc)
        future_ts = datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc)

        snapshots = [_make_snapshot("ACME", price=10.5)]
        # All bars have a future source timestamp
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=10.5,
            source_ts=future_ts,
        )}

        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=cutoff,
        )

        # The 52w high and related fields should be rejected by the framework
        assert result.assembled_count == 0
        assert len(result.inputs) == 0
        assert result.rejected_count + result.insufficient_count > 0
        # Verify diagnostic explains the rejection
        has_diag = any(
            d.diagnostic_type in ("insufficient_history", "field_rejected_lookahead")
            for d in result.diagnostics
        )
        assert has_diag

    def test_future_bar_date_rejected_even_with_allowed_source_timestamp(self):
        """A bar dated after trading_date is lookahead even if sourced before cutoff."""
        cutoff = datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc)
        allowed_source_ts = datetime(2026, 5, 20, 19, 0, tzinfo=timezone.utc)

        snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": [
            DailyBar(
                date="2026-05-21",
                open=10.0,
                high=10.5,
                low=9.8,
                close=10.4,
                volume=100_000,
                split_adjusted_close=10.4,
                adj_close=10.4,
                source_timestamp=allowed_source_ts,
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}

        result = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars=bars,
            trading_date="2026-05-20",
            cutoff_timestamp=cutoff,
        )

        assert result.assembled_count == 0
        assert len(result.inputs) == 0
        assert result.rejected_count == 1
        assert any(
            d.diagnostic_type == "field_rejected_lookahead"
            and (
                "bar.date 2026-05-21 after evidence_session_date 2026-05-20"
                in (d.detail or "")
            )
            for d in result.diagnostics
        )

    def test_future_bar_produces_no_feature_snapshot(self, db_session):
        """End-to-end: future-dated daily bar produces 0 feature snapshots."""
        _setup_canonical_universe(db_session, tickers=["ACME"], prices={"ACME": 10.5})

        cutoff = datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc)
        allowed_source_ts = datetime(2026, 5, 20, 19, 0, tzinfo=timezone.utc)

        bars = {"ACME": [
            DailyBar(
                date="2026-05-21",
                open=10.0,
                high=10.5,
                low=9.8,
                close=10.4,
                volume=100_000,
                split_adjusted_close=10.4,
                adj_close=10.4,
                source_timestamp=allowed_source_ts,
                source_provider="FMP",
                lineage_hash="bar-lineage-hash",
            ),
        ]}
        snapshots = [_make_snapshot("ACME", price=10.5)]

        assembly = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=cutoff,
        )
        assert len(assembly.inputs) == 0

        # Even if we run orchestration, no feature snapshots should be produced
        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert db_session.query(FeatureSnapshot).count() == 0
        assert db_session.query(SignalRegistry).count() == 0


# ===================================================================
# 5. Missing/insufficient history: no zero-fill
# ===================================================================


class TestMissingData:
    def test_no_bars_produces_evidence_session_diagnostic(self):
        """Ticker with no bar history gets an explicit diagnostic, not zero-fill."""
        snapshots = [_make_snapshot("ACME")]
        cutoff = _ts()
        result = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars={},  # No bars at all
            trading_date="2026-05-20",
            cutoff_timestamp=cutoff,
        )
        assert result.assembled_count == 0
        assert result.rejected_count == 1
        assert len(result.inputs) == 0
        diag = result.diagnostics[0]
        assert diag.diagnostic_type == "evidence_session_bar_unavailable"
        assert "ACME" == diag.ticker
        rejected = {field.name: field for field in result.rejected_fields}
        assert rejected["price"].allowed_cutoff == cutoff
        assert rejected["high_52w"].allowed_cutoff == cutoff
        assert rejected["n_sessions_in_window"].allowed_cutoff == cutoff

    def test_missing_price_in_snapshot_does_not_override_evidence_close(self):
        snapshots = [_make_snapshot("ACME", price=None)]
        bars = {"ACME": _make_breakout_bars(n=100)}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.inputs[0].market_data["price"] == 10.5

    def test_zero_evidence_split_adjusted_close_is_not_treated_as_missing(self):
        """0 means observed zero, even though M4 detector later rejects it."""
        snapshots = [_make_snapshot("ACME", price=0.0)]
        bars = {"ACME": _make_breakout_bars(n=100, close_price=0.0)}
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.inputs[0].market_data["price"] == 0.0

    def test_insufficient_history_not_zero_filled(self):
        """Verify that n_sessions_in_window reflects actual sessions, not 252."""
        snapshots = [_make_snapshot("ACME")]
        bars = {"ACME": _make_breakout_bars(n=50)}  # only 50 sessions
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        inp = result.inputs[0]
        assert inp.market_data["n_sessions_in_window"] == 50
        assert any(
            d.diagnostic_type == "short_history"
            for d in result.diagnostics
        )

    def test_mixed_available_and_missing(self):
        """One ticker has bars, another doesn't — both handled correctly."""
        snapshots = [
            _make_snapshot("ACME"),
            _make_snapshot("BETA"),
        ]
        bars = {"ACME": _make_breakout_bars(n=100)}  # BETA has no bars
        result = assemble_m4_daily(
            snapshots=snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert result.assembled_count == 1
        assert result.rejected_count == 1
        assert result.inputs[0].ticker == "ACME"
        assert any(
            d.ticker == "BETA"
            and d.diagnostic_type == "evidence_session_bar_unavailable"
            for d in result.diagnostics
        )


# ===================================================================
# 6. Orchestration integration: assembled inputs by pattern_id
# ===================================================================


class TestOrchestrationWithAssembly:
    def test_assembled_m4_inputs_produce_feature_snapshots(self, db_session):
        """M4 assembled inputs fed through orchestration produce feature_snapshots > 0."""
        tickers = ["ACME"]
        prices = {"ACME": 10.5}
        _, db_snapshots = _setup_canonical_universe(
            db_session, tickers=tickers, prices=prices,
        )

        # Assemble M4 inputs from fixtures
        fixture_snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=10.5,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )
        assert len(assembly.inputs) == 1

        # Run orchestration with assembled inputs
        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        # Acceptance: feature_snapshots > 0
        feat_count = db_session.query(FeatureSnapshot).count()
        assert feat_count > 0, f"Expected feature_snapshots > 0, got {feat_count}"

        # The feature snapshot should have M4 pattern_id
        feat = db_session.query(FeatureSnapshot).first()
        assert feat.pattern_id == "M4"
        assert feat.ticker == "ACME"

    def test_assembled_inputs_signal_fires_on_breakout(self, db_session):
        """M4 should fire a signal when price >= high_52w."""
        _, _ = _setup_canonical_universe(
            db_session, tickers=["ACME"], prices={"ACME": 10.5},
        )

        fixture_snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=10.5,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.metrics["total_signals_persisted"] >= 1
        sig = db_session.query(SignalRegistry).first()
        assert sig is not None
        assert sig.pattern_id == "M4"
        assert sig.ticker == "ACME"

    def test_assembled_inputs_no_signal_below_high(self, db_session):
        """M4 should not fire when price < high_52w."""
        _, _ = _setup_canonical_universe(
            db_session, tickers=["ACME"], prices={"ACME": 9.0},
        )

        fixture_snapshots = [_make_snapshot("ACME", price=9.0)]
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=9.0,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        assert result.metrics["total_signals_persisted"] == 0
        assert db_session.query(FeatureSnapshot).count() == 1

        feat = db_session.query(FeatureSnapshot).one()
        assert feat.pattern_id == "M4"
        assert feat.ticker == "ACME"
        feature_json = json.loads(feat.feature_json)
        assert feature_json["rejection_reason"] == "below_high"
        assert feature_json["signal_generated"] is False
        assert feature_json["H_52w"] == 10.0
        assert feature_json["P_close"] == 9.0

        # Detector evaluated the input (not crashed, not skipped by guard)
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["evaluated_count"] == 1
        assert diag["feature_snapshot_count"] == 1
        assert diag["skipped_count"] == 1  # no signal = skipped

    def test_short_history_below_floor_persists_feature_without_signal(self, db_session):
        """Tiny-history breakouts remain auditable but cannot write M4 signals."""
        _, _ = _setup_canonical_universe(
            db_session, tickers=["ACME"], prices={"ACME": 10.5},
        )

        fixture_snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": _make_breakout_bars(
            n=3, high_52w=10.0, close_price=10.5,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.metrics["total_signals_persisted"] == 0
        assert db_session.query(SignalRegistry).count() == 0
        assert db_session.query(FeatureSnapshot).count() == 1
        feature_json = json.loads(db_session.query(FeatureSnapshot).one().feature_json)
        assert feature_json["signal_generated"] is False
        assert feature_json["rejection_reason"] == "short_history_below_signal_floor"
        assert feature_json["n_sessions_in_window"] == 3
        assert feature_json["short_history_flag"] is True
        assert feature_json["short_history_below_signal_floor"] is True

    def test_assembled_inputs_resolve_lineage_hashes_to_feature_snapshot_ids(
        self, db_session,
    ):
        """Assembled lineage hashes must persist as data_lineage_ids when resolvable."""
        _setup_canonical_universe(
            db_session, tickers=["ACME"], prices={"ACME": 9.0},
        )
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(),
            raw_payload_hash="bar-lineage-hash",
        )

        fixture_snapshots = [_make_snapshot("ACME", price=9.0)]
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=9.0,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        feat = db_session.query(FeatureSnapshot).one()
        assert lineage.data_lineage_id in json.loads(feat.data_lineage_ids)

    def test_no_signal_feature_snapshot_dedupes_on_rerun(self, db_session):
        """Same no-signal feature content should not duplicate feature snapshots."""
        _setup_canonical_universe(
            db_session, tickers=["ACME"], prices={"ACME": 9.0},
        )

        fixture_snapshots = [_make_snapshot("ACME", price=9.0)]
        bars = {"ACME": _make_breakout_bars(
            n=252, high_52w=10.0, close_price=9.0,
        )}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        for _ in range(2):
            orch = DetectorOrchestrationJob(
                db_session,
                detectors=[M4Detector()],
                trading_date="2026-05-20",
                assembled_inputs={"M4": assembly.inputs},
            )
            run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        assert db_session.query(FeatureSnapshot).count() == 1


# ===================================================================
# 7. Orchestration compatibility with existing injected inputs
# ===================================================================


class AlwaysFiresDetector(BasePatternDetector):
    """Test detector that fires on every input."""

    pattern_id = "TEST_FIRES"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        identity = stable_hash({
            "pattern_id": self.pattern_id,
            "ticker": inp.ticker,
            "setup": "fixture",
        })
        features = PatternFeatures(
            features={
                "score": 0.95,
                "price": inp.market_data.get("price", 0),
                "signal_identity_hash": identity,
                "signal_identity_components": {
                    "pattern_id": self.pattern_id,
                    "ticker": inp.ticker,
                    "setup": "fixture",
                },
            },
            feature_manifest_version="test-v1",
            fidelity_tier=FidelityTier.FULL,
            point_in_time_passed=True,
            lookahead_guard_passed=True,
        )
        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=inp.asof_timestamp,
            features=features,
            signals=[PatternSignal(
                direction=SignalDirection.LONG,
                raw_signal_strength=0.95,
                raw_expected_edge=0.05,
                signal_horizon="10d",
            )],
            input_hashes={"market_data": stable_hash(inp.market_data)},
        )


class TestOrchestrationCompatibility:
    def test_injected_inputs_still_work(self, db_session):
        """Existing flat injected-input path remains functional."""
        _setup_canonical_universe(db_session)

        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"price": 5.0},
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            inputs=inputs,
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.ok or result.status == "finished"
        assert result.metrics["total_signals_persisted"] == 1

    def test_assembly_mode_skips_detector_without_assembled_inputs(self, db_session):
        """Assembly mode skips detectors without assembled inputs instead of falling back."""
        _setup_canonical_universe(db_session)

        # Assembled M4 input
        fixture_snapshots = [_make_snapshot("ACME", price=10.5)]
        bars = {"ACME": _make_breakout_bars(n=252, high_52w=10.0, close_price=10.5)}
        assembly = assemble_m4_daily(
            snapshots=fixture_snapshots, daily_bars=bars,
            trading_date="2026-05-20", cutoff_timestamp=_ts(),
        )

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector(), AlwaysFiresDetector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": assembly.inputs},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        # Both detectors are represented, but only M4 has assembly input.
        assert result.metrics["detector_count"] == 2

        diags = {d["detector_id"]: d for d in result.metrics["detector_diagnostics"]}
        assert diags["M4"]["evaluated_count"] == 1
        assert diags["TEST_FIRES"]["evaluated_count"] == 0
        assert diags["TEST_FIRES"]["detector_status"] == "skipped"
        assert diags["TEST_FIRES"]["callable_status"] == "assembly_missing_inputs"

        # Assembly diagnostics show the missing assembler/input path.
        assert any(
            d["detector_id"] == "TEST_FIRES"
            and d["diagnostic"] == "assembled_inputs_missing"
            for d in result.metrics["assembly_diagnostics"]
        )

    def test_no_crash_when_assembled_inputs_empty(self, db_session):
        """Empty assembled inputs for a pattern should not crash orchestration."""
        _setup_canonical_universe(db_session)

        orch = DetectorOrchestrationJob(
            db_session,
            detectors=[M4Detector()],
            trading_date="2026-05-20",
            assembled_inputs={"M4": []},
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        # Should not crash, just produce no signals
        assert result.status == "finished"
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["evaluated_count"] == 0
        assert diag["detector_status"] == "skipped"
        assert diag["callable_status"] == "assembly_empty_inputs"
        assert result.metrics["assembly_diagnostics"] == [{
            "detector_id": "M4",
            "diagnostic": "assembled_inputs_empty",
        }]


# ===================================================================
# 8. Assembly registry integration: diagnostics for missing assemblers
# ===================================================================


class TestRegistryDiagnostics:
    def test_detector_only_pattern_reports_diagnostic(self):
        """Patterns with detectors but no assembler produce explicit diagnostic."""
        registry = AssemblyRegistry(assemblers={"M4": assemble_m4_daily})
        entry = registry.get("M1")
        assert entry.status == AssemblerStatus.DETECTOR_ONLY
        assert entry.assembler is None

    def test_reserved_pattern_reports_diagnostic(self):
        """Reserved patterns with no detector or assembler report reserved status."""
        registry = AssemblyRegistry()
        entry = registry.get("I2")
        assert entry.status == AssemblerStatus.RESERVED
        assert entry.assembler is None

    def test_full_diagnostics_map(self):
        """The full diagnostics map shows all 17 patterns."""
        registry = AssemblyRegistry(assemblers={"M4": assemble_m4_daily})
        diag = registry.diagnostics()
        assert diag == {
            "M1": "detector_only",
            "M2": "detector_only",
            "M3": "detector_only",
            "M4": "implemented",
            "M5": "detector_only",
            "M6": "detector_only",
            "M7": "detector_only",
            "I1": "detector_only",
            "I2": "reserved",
            "I3": "reserved",
            "I4": "reserved",
            "I5": "reserved",
            "I6": "reserved",
            "I7": "reserved",
            "I8": "detector_only",
            "I9": "reserved",
            "I10": "reserved",
        }
