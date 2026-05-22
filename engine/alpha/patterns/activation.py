"""Shared helpers for watchlist-to-activation detector plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SameSessionFreshness:
    """Freshness proof for one-session watchlists."""

    source_freshness_passed: bool
    watchlist_identity_passed: bool
    watchlist_session_match: bool
    signal_freshness_passed: bool


@dataclass(frozen=True)
class ExpiringWatchlistFreshness:
    """Freshness proof for multi-session watchlists with decay."""

    source_freshness_passed: bool
    watchlist_identity_passed: bool
    watchlist_session_match: bool
    signal_freshness_passed: bool
    decay_weight: float
    age_sessions: Optional[int]


def is_present(value: Any) -> bool:
    """Return True only for non-empty values; whitespace strings are missing."""
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def required_fields_present(data: Mapping[str, Any], fields: Sequence[str]) -> bool:
    """Return True when all required fields are present and non-empty."""
    return all(is_present(data.get(field)) for field in fields)


def parse_session_date(value: Any) -> Optional[date]:
    """Parse YYYY-MM-DD or timestamp-like values into a date."""
    if not is_present(value):
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def same_session_freshness(
    data: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
    valid_session_field: str,
    activation_session_field: str,
    source_freshness_field: str = "signal_freshness_passed",
) -> SameSessionFreshness:
    """Build freshness proof for a watchlist that can activate only in one session."""
    source_freshness_passed = data.get(source_freshness_field) is True
    watchlist_identity_passed = required_fields_present(data, identity_fields)
    watchlist_valid_session = data.get(valid_session_field)
    activation_session = data.get(activation_session_field)
    watchlist_session_match = (
        is_present(watchlist_valid_session)
        and is_present(activation_session)
        and str(watchlist_valid_session).strip() == str(activation_session).strip()
    )
    return SameSessionFreshness(
        source_freshness_passed=source_freshness_passed,
        watchlist_identity_passed=watchlist_identity_passed,
        watchlist_session_match=watchlist_session_match,
        signal_freshness_passed=(
            source_freshness_passed
            and watchlist_identity_passed
            and watchlist_session_match
        ),
    )


def expiring_watchlist_freshness(
    data: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
    expiration_session_field: str,
    activation_session_field: str,
    age_field: str,
    decay_by_age: Mapping[int, float],
    source_freshness_field: str = "signal_freshness_passed",
) -> ExpiringWatchlistFreshness:
    """Build freshness proof for a watchlist that can activate over multiple sessions."""
    source_freshness_passed = data.get(source_freshness_field) is True
    watchlist_identity_passed = required_fields_present(data, identity_fields)
    watchlist_expiration_session = parse_session_date(data.get(expiration_session_field))
    activation_session_date = parse_session_date(data.get(activation_session_field))
    age_sessions = None
    try:
        if data.get(age_field) is not None:
            age_sessions = int(data[age_field])
    except (TypeError, ValueError):
        age_sessions = None
    decay_weight = decay_by_age.get(age_sessions, 0.0) if age_sessions is not None else 0.0
    watchlist_session_match = (
        watchlist_expiration_session is not None
        and activation_session_date is not None
        and activation_session_date <= watchlist_expiration_session
        and decay_weight > 0
    )
    return ExpiringWatchlistFreshness(
        source_freshness_passed=source_freshness_passed,
        watchlist_identity_passed=watchlist_identity_passed,
        watchlist_session_match=watchlist_session_match,
        signal_freshness_passed=(
            source_freshness_passed
            and watchlist_identity_passed
            and watchlist_session_match
        ),
        decay_weight=decay_weight,
        age_sessions=age_sessions,
    )
