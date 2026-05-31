from __future__ import annotations

from datetime import date, datetime, timezone
import os
from unittest.mock import MagicMock

import pytest
import requests

from alpha.data.nasdaq import (
    NasdaqListingStatus,
    NasdaqTraderListingAdapter,
    _parse_feed_datetime,
    _status_from_halt_events,
)


def _mock_response(status_code: int = 200, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


def _nasdaq_directory(rows, *, footer="File Creation Time: 0529202621:31|||||||"):
    header = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
    body = [
        f"{symbol}|{name}|Q|N|N|100|N|N"
        for symbol, name in rows
    ]
    return "\n".join([header, *body, footer])


def _nasdaq_directory_full(rows, *, footer="File Creation Time: 0529202621:31|||||||"):
    header = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
    body = []
    for row in rows:
        if len(row) == 2:
            symbol, name = row
            test_issue = "N"
            etf = "N"
        else:
            symbol, name, test_issue, etf = row
        body.append(f"{symbol}|{name}|Q|{test_issue}|N|100|{etf}|N")
    return "\n".join([header, *body, footer])


def _other_directory(rows, *, footer="File Creation Time: 0529202621:31||||||"):
    header = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"
    body = [
        f"{symbol}|{name}|N|{symbol}|N|100|N|{symbol}"
        for symbol, name in rows
    ]
    return "\n".join([header, *body, footer])


def _adds_deletes(rows=(), *, footer="File Creation Time: 0529202621:32|||||"):
    header = "Symbol|Company Name|NASDAQ Action|BX Action|PSX Action|Effective Date|Primary Listing Market"
    body = [
        f"{symbol}|{name}|{nasdaq_action}|{bx_action}|{psx_action}|{effective_date}|{market}"
        for symbol, name, nasdaq_action, bx_action, psx_action, effective_date, market in rows
    ]
    return "\n".join([header, *body, footer])


def _halt_rss(items=()):
    body = []
    for item in items:
        if len(item) == 4:
            symbol, reason, halt_date, halt_time = item
            item_pub = "Fri, 29 May 2026 04:00:00 GMT"
        else:
            symbol, reason, halt_date, halt_time, item_pub = item
        body.append(
            f"""
      <item>
        <title>{symbol}</title>
        <pubDate>{item_pub}</pubDate>
        <ndaq:IssueSymbol>{symbol}</ndaq:IssueSymbol>
        <ndaq:IssueName>{symbol} Test Common Stock</ndaq:IssueName>
        <ndaq:Mkt>Q</ndaq:Mkt>
        <ndaq:ReasonCode>{reason}</ndaq:ReasonCode>
        <ndaq:HaltDate>{halt_date}</ndaq:HaltDate>
        <ndaq:HaltTime>{halt_time}</ndaq:HaltTime>
        <ndaq:ResumptionDate />
        <ndaq:ResumptionTradeTime />
      </item>"""
        )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/">
  <channel>
    <title>NASDAQTrader.com</title>
    <pubDate>Sun, 31 May 2026 10:00:00 GMT</pubDate>
    <ndaq:numItems>{len(items)}</ndaq:numItems>
    {''.join(body)}
  </channel>
</rss>"""


def _rows(prefix, count):
    return [(f"{prefix}{idx:04d}", f"{prefix} filler {idx}") for idx in range(count)]


def _adapter_for(*texts):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [_mock_response(text=text) for text in texts]
    return NasdaqTraderListingAdapter(session=session), session


ASOF_AFTER_FILE = datetime(2026, 5, 30, 3, 0, tzinfo=timezone.utc)


class TestNasdaqTraderListingAdapter:
    def test_present_symbol_is_listed_active(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
        )

        resp = adapter.get_listing_status("aapl", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert resp.data.matched_symbol == "AAPL"
        assert resp.data.pit_knowable_at_asof is True

    def test_absent_symbol_is_inconclusive_not_delisted_for_known_residuals(self):
        for symbol in ["GLTO", "LITM", "OTH", "TTSH", "VBIX", "VVPR", "ZGM"]:
            adapter, _ = _adapter_for(
                _nasdaq_directory(_rows("N", 1001)),
                _other_directory(_rows("O", 1001)),
                _adds_deletes(),
                _halt_rss(),
            )

            resp = adapter.get_listing_status(symbol, asof=ASOF_AFTER_FILE)

            assert resp.ok
            assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
            assert resp.data.status is not NasdaqListingStatus.DELISTED
            assert resp.data.reason == "symbol_absent_from_current_snapshot"

    def test_adds_deletes_delete_is_delisted(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes([
                ("CVGW", "Calavo Growers, Inc.", "Delete", "Delete", "Delete", "5/29/2026", "Q")
            ]),
        )

        resp = adapter.get_listing_status("CVGW", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trading_system_adds_deletes_delete"

    def test_reason_d_halt_is_delisted_but_ludp_is_not(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss([
                ("DEAD", "D", "05/29/2026", "19:50:00.000"),
                ("QH", "LUDP", "05/29/2026", "09:30:39.140"),
            ]),
        )

        dead = adapter.get_listing_status("DEAD", asof=ASOF_AFTER_FILE)

        assert dead.ok
        assert dead.data.status is NasdaqListingStatus.DELISTED
        assert dead.data.reason == "trade_halt_reason_D_security_deletion"

        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss([("QH", "LUDP", "05/29/2026", "09:30:39.140")]),
        )
        ludp = adapter.get_listing_status("QH", asof=ASOF_AFTER_FILE)
        assert ludp.ok
        assert ludp.data.status is NasdaqListingStatus.INCONCLUSIVE

    def test_halt_rss_mojibake_bom_from_live_shape_is_parsed(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            "ï»¿" + _halt_rss([("DEAD", "D", "05/29/2026", "19:50:00.000")]),
        )

        resp = adapter.get_listing_status("DEAD", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED

    def test_symbol_normalization_matches_class_share_punctuation(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("BRK-B", "Berkshire Hathaway Class B"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
        )

        resp = adapter.get_listing_status(" brk.b ", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert resp.data.matched_symbol == "BRK-B"

    def test_compact_variant_does_not_cross_match_distinct_issuer(self):
        for query, compact in (("NE.A", "NEA"), ("CTV.B", "CTVB")):
            adapter, _ = _adapter_for(
                _nasdaq_directory([(compact, "Different Listed Issuer Common Stock"), *_rows("N", 1000)]),
                _other_directory(_rows("O", 1001)),
                _adds_deletes(),
                _halt_rss(),
            )

            resp = adapter.get_listing_status(query, asof=ASOF_AFTER_FILE)

            assert resp.ok
            assert resp.data.status is not NasdaqListingStatus.LISTED_ACTIVE
            assert resp.data.matched_symbol != compact

    def test_unit_suffix_compact_bridge_still_matches_same_issuer_unit(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("XU", "Example Acquisition Corp. - Units"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
        )

        resp = adapter.get_listing_status("X.U", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert resp.data.matched_symbol == "XU"

    def test_unit_suffix_compact_bridge_does_not_match_common_stock_ending_u(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("ACIU", "AC Immune SA - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("ACI.U", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.status is not NasdaqListingStatus.LISTED_ACTIVE

    def test_normalization_does_not_confuse_common_with_unit_or_warrant(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AACIU", "Armada Acquisition Corp. III - Units"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("AACI", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE

    def test_test_issue_etf_and_noncommon_rows_are_not_listed_active(self):
        cases = [
            ("ZVZZT", "NASDAQ TEST STOCK", "Y", "N"),
            ("ETFZ", "Example ETF", "N", "Y"),
            ("WRTZ", "Example Co. - Warrants", "N", "N"),
        ]
        for symbol, name, test_issue, etf in cases:
            adapter, _ = _adapter_for(
                _nasdaq_directory_full([(symbol, name, test_issue, etf), *_rows("N", 1000)]),
                _other_directory(_rows("O", 1001)),
                _adds_deletes(),
                _halt_rss(),
            )

            resp = adapter.get_listing_status(symbol, asof=ASOF_AFTER_FILE)

            assert resp.ok
            assert resp.data.status is not NasdaqListingStatus.LISTED_ACTIVE

    def test_missing_trailer_is_unavailable(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001), footer=""),
        )

        resp = adapter.get_listing_status("AAPL", asof=ASOF_AFTER_FILE)

        assert not resp.ok
        assert resp.data.status is NasdaqListingStatus.UNAVAILABLE
        assert resp.error.error_type == "parse"
        assert "File Creation Time" in resp.error.message

    def test_malformed_trailer_is_unavailable(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001), footer="File Creation Time: bad|||||||"),
        )

        resp = adapter.get_listing_status("AAPL", asof=ASOF_AFTER_FILE)

        assert not resp.ok
        assert resp.data.status is NasdaqListingStatus.UNAVAILABLE
        assert resp.error.error_type == "parse"
        assert "Malformed" in resp.error.message

    def test_truncated_directory_is_unavailable(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock")]),
        )

        resp = adapter.get_listing_status("AAPL", asof=ASOF_AFTER_FILE)

        assert not resp.ok
        assert resp.data.status is NasdaqListingStatus.UNAVAILABLE
        assert "below minimum" in resp.error.message

    def test_provider_error_is_unavailable(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.Timeout()
        adapter = NasdaqTraderListingAdapter(session=session)

        resp = adapter.get_listing_status("AAPL", asof=ASOF_AFTER_FILE)

        assert not resp.ok
        assert resp.data.status is NasdaqListingStatus.UNAVAILABLE
        assert resp.error.error_type == "timeout"

    def test_live_snapshot_before_asof_is_inconclusive_not_confident_status(self):
        before_file_creation = datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc)
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
        )

        resp = adapter.get_listing_status("AAPL", asof=before_file_creation)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.pit_knowable_at_asof is False
        assert resp.data.reason == "directory_match_not_knowable_at_asof"

    def test_delisted_years_ago_before_deletion_asof_is_inconclusive(self):
        before_deletion = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes([
                ("CVGW", "Calavo Growers, Inc.", "Delete", "Delete", "Delete", "5/29/2026", "Q")
            ]),
            _halt_rss([("CVGW", "D", "05/29/2026", "19:50:00.000")]),
        )

        resp = adapter.get_listing_status("CVGW", asof=before_deletion)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.pit_knowable_at_asof is False
        assert resp.data.reason == "adds_deletes_delete_not_knowable_at_asof"

    def test_halt_after_asof_is_inconclusive_not_delisted(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(_rows("N", 1001)),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss([(
                "DEAD",
                "D",
                "05/29/2026",
                "19:50:00.000",
                "Fri, 29 May 2026 23:59:00 GMT",
            )]),
        )

        resp = adapter.get_listing_status(
            "DEAD",
            asof=datetime(2026, 5, 29, 23, 51, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.pit_knowable_at_asof is False
        assert resp.data.reason == "halt_reason_D_not_knowable_at_asof"

    def test_duplicate_header_body_row_is_unavailable(self):
        header = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
        adapter, _ = _adapter_for(
            "\n".join([
                header,
                header,
                *[
                    f"N{idx:04d}|N filler {idx}|Q|N|N|100|N|N"
                    for idx in range(1001)
                ],
                "File Creation Time: 0529202621:31|||||||",
            ])
        )

        resp = adapter.get_listing_status("AAPL", asof=ASOF_AFTER_FILE)

        assert not resp.ok
        assert resp.data.status is NasdaqListingStatus.UNAVAILABLE
        assert resp.error.error_type == "parse"
        assert "Duplicate Nasdaq header" in resp.error.message

    def test_batch_reports_inconclusive_rate_telemetry(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_statuses(["AAPL", "ZZZZNOTREAL"], asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.lineage.data_quality_flags["symbol_count"] == 2
        assert resp.lineage.data_quality_flags["inconclusive_count"] == 1
        assert resp.lineage.data_quality_flags["inconclusive_rate"] == 0.5

    def test_archive_capture_and_precapture_inconclusive(self, db_session):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
        )

        captured = adapter.archive_current_snapshot(db_session, asof=ASOF_AFTER_FILE)

        assert captured.ok
        assert captured.data.inserted_snapshots == 3
        assert captured.data.inserted_rows == 2002

        before_capture = datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc)
        precapture = adapter.get_listing_status(
            "AAPL",
            asof=before_capture,
            archive_session=db_session,
            use_live=False,
        )
        assert precapture.ok
        assert precapture.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert precapture.data.reason == "no_archived_snapshot_for_asof"
        assert precapture.data.pit_knowable_at_asof is False

        archived = adapter.get_listing_status(
            "AAPL",
            asof=ASOF_AFTER_FILE,
            archive_session=db_session,
            use_live=False,
        )
        assert archived.ok
        assert archived.data.status is NasdaqListingStatus.LISTED_ACTIVE

        stale = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2027, 1, 1, tzinfo=timezone.utc),
            archive_session=db_session,
            use_live=False,
        )
        assert stale.ok
        assert stale.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert stale.data.pit_knowable_at_asof is False
        assert stale.data.reason == "archived_snapshot_stale_for_asof"

    def test_feed_timestamp_is_interpreted_as_eastern_time(self):
        parsed = _parse_feed_datetime("05/29/2026", "19:50:00.000")

        assert parsed == datetime(2026, 5, 29, 23, 50, tzinfo=timezone.utc)


def test_live_nasdaq_smoke():
    if os.environ.get("ALPHA_LIVE_NASDAQ_SMOKE") != "1":
        pytest.skip("set ALPHA_LIVE_NASDAQ_SMOKE=1 to run live Nasdaq smoke")

    adapter = NasdaqTraderListingAdapter(user_agent="AlphaCapital live Nasdaq smoke")
    asof = datetime.now(timezone.utc)

    listed = adapter.get_listing_status("AAPL", asof=asof)
    assert listed.ok
    assert listed.data.status is NasdaqListingStatus.LISTED_ACTIVE

    absent = adapter.get_listing_status("ZZZZNOTREAL", asof=asof)
    assert absent.ok
    assert absent.data.status is NasdaqListingStatus.INCONCLUSIVE

    halt_resp = adapter.get_halt_events(
        asof=datetime(2026, 5, 30, 3, 0, tzinfo=timezone.utc),
        halt_date=date(2026, 5, 27),
    )
    assert halt_resp.ok
    reason_d = _status_from_halt_events(
        "CVGW",
        datetime(2026, 5, 30, 3, 0, tzinfo=timezone.utc),
        halt_resp.data,
    )
    assert reason_d is not None
    assert reason_d.status is NasdaqListingStatus.DELISTED
