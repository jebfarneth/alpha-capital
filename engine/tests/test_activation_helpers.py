from alpha.patterns.activation import (
    expiring_watchlist_freshness,
    is_present,
    parse_session_date,
    required_fields_present,
    same_session_freshness,
)


def test_is_present_rejects_blank_strings():
    assert is_present("abc") is True
    assert is_present("  ") is False
    assert is_present(None) is False


def test_required_fields_present_rejects_missing_or_blank_fields():
    data = {"activation_id": "act-1", "activation_timestamp": "2026-05-20T15:00:00Z"}
    assert required_fields_present(data, ("activation_id", "activation_timestamp")) is True
    assert required_fields_present({**data, "activation_id": " "}, ("activation_id", "activation_timestamp")) is False
    assert required_fields_present({"activation_id": "act-1"}, ("activation_id", "activation_timestamp")) is False


def test_parse_session_date_accepts_dates_and_timestamp_prefixes():
    assert parse_session_date("2026-05-20") is not None
    assert parse_session_date("2026-05-20T15:00:00Z") is not None
    assert parse_session_date("2026-05-20T15:00:00-04:00") is not None
    assert parse_session_date("not-a-date") is None
    assert parse_session_date(" ") is None
    assert parse_session_date(None) is None


def test_same_session_freshness_requires_true_source_identity_and_session_match():
    data = {
        "signal_freshness_passed": True,
        "watchlist_signal_id": "watch-1",
        "watchlist_scan_date": "2026-05-19",
        "watchlist_valid_session": "2026-05-20",
        "activation_session": "2026-05-20",
    }
    freshness = same_session_freshness(
        data,
        identity_fields=("watchlist_signal_id", "watchlist_scan_date", "watchlist_valid_session", "activation_session"),
        valid_session_field="watchlist_valid_session",
        activation_session_field="activation_session",
    )
    assert freshness.signal_freshness_passed is True

    freshness = same_session_freshness(
        {**data, "signal_freshness_passed": "true"},
        identity_fields=("watchlist_signal_id", "watchlist_scan_date", "watchlist_valid_session", "activation_session"),
        valid_session_field="watchlist_valid_session",
        activation_session_field="activation_session",
    )
    assert freshness.source_freshness_passed is False
    assert freshness.signal_freshness_passed is False


def test_expiring_watchlist_freshness_applies_age_decay_and_expiration():
    data = {
        "signal_freshness_passed": True,
        "watchlist_signal_id": "watch-1",
        "watchlist_scan_date": "2026-05-19",
        "watchlist_expiration_session": "2026-05-22",
        "activation_session": "2026-05-21",
        "watchlist_age_sessions": 2,
    }
    freshness = expiring_watchlist_freshness(
        data,
        identity_fields=(
            "watchlist_signal_id",
            "watchlist_scan_date",
            "watchlist_expiration_session",
            "activation_session",
            "watchlist_age_sessions",
        ),
        expiration_session_field="watchlist_expiration_session",
        activation_session_field="activation_session",
        age_field="watchlist_age_sessions",
        decay_by_age={1: 1.0, 2: 0.85, 3: 0.70},
    )
    assert freshness.signal_freshness_passed is True
    assert freshness.decay_weight == 0.85
    assert freshness.age_sessions == 2

    expired = expiring_watchlist_freshness(
        {**data, "activation_session": "2026-05-23", "watchlist_age_sessions": 4},
        identity_fields=(
            "watchlist_signal_id",
            "watchlist_scan_date",
            "watchlist_expiration_session",
            "activation_session",
            "watchlist_age_sessions",
        ),
        expiration_session_field="watchlist_expiration_session",
        activation_session_field="activation_session",
        age_field="watchlist_age_sessions",
        decay_by_age={1: 1.0, 2: 0.85, 3: 0.70},
    )
    assert expired.watchlist_session_match is False
    assert expired.signal_freshness_passed is False
    assert expired.decay_weight == 0.0
