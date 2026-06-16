"""Market session calendar helpers.

The engine needs completed-session semantics, not just calendar dates.
This module implements the U.S. equity regular-session calendar rules the
feature assembly path depends on: weekends, NYSE holidays, and audited
09:30-16:00 or early-close America/New_York session hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Any

EASTERN_TZ = ZoneInfo("America/New_York")
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)
MARKET_EARLY_CLOSE_ET = time(13, 0)
_NYSE_EARLY_CLOSES: dict[int, frozenset[date]] = {
    2026: frozenset({
        date(2026, 11, 27),
        date(2026, 12, 24),
    }),
    2027: frozenset({
        date(2027, 11, 26),
    }),
    2028: frozenset({
        date(2028, 7, 3),
        date(2028, 11, 24),
    }),
}


@dataclass(frozen=True)
class SessionResolution:
    """Resolved market-session semantics for one engine run timestamp."""

    decision_date: str
    evidence_session_date: str
    next_execution_session: str
    is_premarket_decision_window: bool


def resolve_us_equity_session(run_timestamp: datetime) -> SessionResolution:
    """Resolve decision/evidence/execution sessions for a run timestamp.

    The decision date is the local New York calendar date of the run. The
    evidence session is the last completed regular U.S. equity session. The
    next execution session is the regular session that can next accept orders.
    """
    if run_timestamp.tzinfo is None or run_timestamp.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware")

    local_ts = run_timestamp.astimezone(EASTERN_TZ)
    decision_day = local_ts.date()
    is_session_day = is_us_equity_session(decision_day)
    local_time = local_ts.time()
    session_close = (
        us_equity_session_close_time(decision_day)
        if is_session_day else MARKET_CLOSE_ET
    )

    if is_session_day and local_time >= session_close:
        evidence_day = decision_day
    else:
        evidence_day = previous_us_equity_session(decision_day)

    is_premarket = is_session_day and local_time < MARKET_OPEN_ET
    if is_session_day and local_time < session_close:
        next_execution_day = decision_day
    else:
        next_execution_day = next_us_equity_session(decision_day + timedelta(days=1))

    return SessionResolution(
        decision_date=decision_day.isoformat(),
        evidence_session_date=evidence_day.isoformat(),
        next_execution_session=next_execution_day.isoformat(),
        is_premarket_decision_window=is_premarket,
    )


def is_us_equity_session(day: date) -> bool:
    """Return True when `day` is a regular NYSE trading session."""
    day = _coerce_session_date(day)
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def nyse_early_closes(year: int) -> set[date]:
    """Curated 13:00 ET NYSE equity early closes for audited years."""

    early_closes = set(_NYSE_EARLY_CLOSES.get(year, frozenset()))
    holidays = nyse_holidays(year)
    invalid = [
        day
        for day in early_closes
        if day.year != year or day in holidays or not is_us_equity_session(day)
    ]
    if invalid:
        bad = ", ".join(day.isoformat() for day in sorted(invalid))
        raise ValueError(f"invalid NYSE early-close date(s): {bad}")
    return early_closes


def previous_us_equity_session(day: date) -> date:
    """Return the regular NYSE session strictly before `day`."""
    day = _coerce_session_date(day)
    cursor = day - timedelta(days=1)
    while not is_us_equity_session(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_us_equity_session(day: date) -> date:
    """Return the first regular NYSE session on or after `day`."""
    day = _coerce_session_date(day)
    cursor = day
    while not is_us_equity_session(cursor):
        cursor += timedelta(days=1)
    return cursor


def nth_us_equity_session(day: date, n: int) -> date:
    """Return the nth regular session on or after `day`.

    `n=1` returns the first regular session on or after `day`. This is useful
    for time barriers that count the entry session as day one.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    day = _coerce_session_date(day)
    cursor = next_us_equity_session(day)
    for _ in range(n - 1):
        cursor = next_us_equity_session(cursor + timedelta(days=1))
    return cursor


def us_equity_session_close_timestamp(day: date) -> datetime:
    """Return the regular-session close timestamp for `day` in UTC.

    Uses audited early-close times where known and degrades future unknown
    years to the regular 16:00 ET close.
    """
    day = _coerce_session_date(day)
    if not is_us_equity_session(day):
        raise ValueError(f"{day.isoformat()} is not a regular U.S. equity session")
    return datetime.combine(
        day, us_equity_session_close_time(day), EASTERN_TZ
    ).astimezone(ZoneInfo("UTC"))


def us_equity_session_close_time(day: date) -> time:
    """Return the day-specific NYSE equity close time in America/New_York."""

    day = _coerce_session_date(day)
    if not is_us_equity_session(day):
        raise ValueError(f"{day.isoformat()} is not a regular U.S. equity session")
    if day in nyse_early_closes(day.year):
        return MARKET_EARLY_CLOSE_ET
    return MARKET_CLOSE_ET


def us_equity_session_open_timestamp(day: date) -> datetime:
    """Return the regular-session open timestamp for `day` in UTC."""
    day = _coerce_session_date(day)
    if not is_us_equity_session(day):
        raise ValueError(f"{day.isoformat()} is not a regular U.S. equity session")
    return datetime.combine(day, MARKET_OPEN_ET, EASTERN_TZ).astimezone(ZoneInfo("UTC"))


def nyse_holidays(year: int) -> set[date]:
    """Regular NYSE holidays for modern U.S. equity sessions.

    This intentionally covers the recurring holidays the engine needs for
    production session resolution. It does not attempt to model unscheduled
    closures.
    """
    holidays = {
        _observed_fixed_holiday(year, 1, 1),       # New Year's Day
        _nth_weekday(year, 1, 0, 3),              # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),              # Washington's Birthday
        _good_friday(year),
        _last_weekday(year, 5, 0),                # Memorial Day
        _observed_fixed_holiday(year, 7, 4),      # Independence Day
        _nth_weekday(year, 9, 0, 1),              # Labor Day
        _nth_weekday(year, 11, 3, 4),             # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),    # Christmas
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))  # Juneteenth
    return {holiday for holiday in holidays if holiday.year == year}


def _coerce_session_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise RuntimeError(
                "invalid U.S. equity session date; expected ISO YYYY-MM-DD, "
                f"got {value!r}"
            ) from exc
    raise RuntimeError(
        "invalid U.S. equity session date; expected date or ISO YYYY-MM-DD, "
        f"got {type(value).__name__}"
    )


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cursor = date(year, month, 1)
    while cursor.weekday() != weekday:
        cursor += timedelta(days=1)
    return cursor + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _good_friday(year: int) -> date:
    return _easter_sunday(year) - timedelta(days=2)


def _easter_sunday(year: int) -> date:
    """Computus for Gregorian Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
