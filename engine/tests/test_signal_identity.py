"""
Detector signal identity regression tests.

Identities are for downstream dedup and must be stable across refreshes of
the same underlying event/setup while changing for genuinely new events.
"""

from __future__ import annotations

from datetime import timedelta

from alpha.patterns.contracts import PatternInput
from alpha.patterns.i1 import I1Detector
from alpha.patterns.i8 import I8Detector
from alpha.patterns.m1 import M1Detector
from alpha.patterns.m2 import M2Detector
from alpha.patterns.m3 import M3Detector
from alpha.patterns.m4 import M4Detector
from alpha.patterns.m5 import M5Detector
from alpha.patterns.m6 import M6Detector
from tests.test_i1 import _confirmed_gap_data, _ts as _i1_ts
from tests.test_i8 import _firing_data as _i8_data
from tests.test_i8 import _ts as _i8_ts
from tests.test_m1 import _firing_data as _m1_data
from tests.test_m1 import _ts as _m1_ts
from tests.test_m2 import _firing_data as _m2_data
from tests.test_m2 import _ts as _m2_ts
from tests.test_m3 import _firing_data as _m3_data
from tests.test_m3 import _ts as _m3_ts
from tests.test_m4 import _m4_base_data, _ts as _m4_ts
from tests.test_m5 import _activation_data as _m5_data
from tests.test_m5 import _ts as _m5_ts
from tests.test_m6 import _firing_market_data as _m6_data
from tests.test_m6 import _ts as _m6_ts


def _identity(result):
    return result.features.features.get("signal_identity_hash")


def test_m1_identity_tracks_earnings_event_not_scan_time():
    det = M1Detector()
    d1 = _m1_data()
    d1["earnings_event_id"] = "ACME-2026Q1"
    d2 = dict(d1)
    d2["delta_t_trading_days"] = 1

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m1_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m1_ts() + timedelta(days=1), market_data=d2, lineage_hashes=["h"]))

    d3 = dict(d1)
    d3["earnings_event_id"] = "ACME-2026Q2"
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m1_ts(), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_m2_standard_identity_alias_uses_accessions_not_delivery_source():
    det = M2Detector()
    d1 = _m2_data()
    d2 = dict(d1)
    d2["source_authority"] = "fmp_backfill"
    d3 = dict(d1)
    d3["sec_accession_numbers"] = ["0001234567-26-000999"]

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m2_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m2_ts() + timedelta(days=1), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m2_ts(), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == r1.features.features["m2_cluster_signature_hash"]
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)
    assert r1.features.features["signal_identity_source"] == "sec_accession_cluster"


def test_m3_identity_only_when_upstream_snapshot_id_exists():
    det = M3Detector()
    no_snapshot = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m3_ts(), market_data=_m3_data(), lineage_hashes=["h"]))
    assert no_snapshot.has_signal
    assert "signal_identity_hash" not in no_snapshot.features.features

    d1 = _m3_data()
    d1["sector_rank_snapshot_id"] = "FMP-SECTOR-RANK-2026-05-20"
    d2 = dict(d1)
    d3 = dict(d1)
    d3["sector_rank_snapshot_id"] = "FMP-SECTOR-RANK-2026-05-21"

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m3_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m3_ts() + timedelta(days=1), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m3_ts(), market_data=d3, lineage_hashes=["h"]))

    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_m4_identity_tracks_high_setup_not_breakout_price():
    det = M4Detector()
    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m4_ts(), market_data=_m4_base_data(price=11.0, high_52w=10.0), lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m4_ts() + timedelta(days=1), market_data=_m4_base_data(price=12.0, high_52w=10.0), lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m4_ts(), market_data=_m4_base_data(price=12.0, high_52w=10.5), lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_m5_identity_tracks_watchlist_setup_not_activation_refresh():
    det = M5Detector()
    d1 = _m5_data()
    d2 = dict(d1)
    d2["activation_id"] = "m5-act-ACME-20260520-151000"
    d2["activation_timestamp"] = "2026-05-20T15:10:00Z"
    d3 = dict(d1)
    d3["watchlist_signal_id"] = "m5-watchlist-ACME-second-setup"

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m5_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m5_ts(), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m5_ts(), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_m6_identity_tracks_compression_setup_not_activation_refresh():
    det = M6Detector()
    d1 = _m6_data()
    d2 = dict(d1)
    d2["activation_id"] = "m6-act-ACME-20260520-151000"
    d2["activation_timestamp"] = "2026-05-20T15:10:00Z"
    d3 = dict(d1)
    d3["watchlist_signal_id"] = "m6-watchlist-ACME-second-setup"

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m6_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m6_ts(), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_m6_ts(), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_i1_identity_tracks_session_gap_not_evaluation_timestamp():
    det = I1Detector()
    d1 = _confirmed_gap_data()
    d2 = dict(d1)
    d2["evaluation_timestamp"] = "2026-05-15T14:10:00Z"
    d2["data_cutoff_timestamp"] = "2026-05-15T14:00:00Z"
    d3 = dict(d1)
    d3["evaluation_timestamp"] = "2026-05-16T14:01:00Z"
    d3["data_cutoff_timestamp"] = "2026-05-16T14:00:00Z"

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i1_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i1_ts(), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i1_ts() + timedelta(days=1), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)


def test_i8_identity_tracks_opening_range_session_not_eval_timestamp():
    det = I8Detector()
    d1 = _i8_data()
    d2 = dict(d1)
    d2["breakout_eval_timestamp"] = "2026-05-15T14:25:00Z"
    d2["data_cutoff_timestamp"] = "2026-05-15T14:25:00Z"
    d3 = dict(d1)
    d3["opening_bar_close_timestamp"] = "2026-05-16T14:00:00Z"
    d3["breakout_eval_timestamp"] = "2026-05-16T14:18:00Z"
    d3["data_cutoff_timestamp"] = "2026-05-16T14:18:00Z"

    r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i8_ts(), market_data=d1, lineage_hashes=["h"]))
    r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i8_ts(), market_data=d2, lineage_hashes=["h"]))
    r3 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_i8_ts() + timedelta(days=1), market_data=d3, lineage_hashes=["h"]))

    assert r1.has_signal and r2.has_signal and r3.has_signal
    assert _identity(r1) == _identity(r2)
    assert _identity(r1) != _identity(r3)
