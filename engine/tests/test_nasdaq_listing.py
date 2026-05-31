from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
import requests

from alpha.data.nasdaq import (
    ADDS_DELETES,
    HALT_RSS,
    NASDAQ_LISTED,
    OTHER_LISTED,
    NasdaqListingStatus,
    NasdaqTraderListingAdapter,
    _parse_feed_datetime,
    _status_from_halt_events,
)
from alpha.db.models import NasdaqListingSnapshot, NasdaqListingSnapshotRow


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


def _halt_rss(items=(), *, channel_pub="Sun, 31 May 2026 10:00:00 GMT"):
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
    <pubDate>{channel_pub}</pubDate>
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


def _insert_archive_snapshot(db_session, source_type, source_ts, rows):
    snapshot = NasdaqListingSnapshot(
        source_type=source_type,
        source_url=f"https://example.test/{source_type}",
        source_knowledge_timestamp=source_ts,
        raw_payload_hash=f"{source_type}-{source_ts.isoformat()}-{len(rows)}",
        raw_payload="scratch payload",
        row_count=len(rows),
        data_quality_flags_json=json.dumps({"source": source_type}),
    )
    db_session.add(snapshot)
    db_session.flush()
    for row in rows:
        row.snapshot_id = snapshot.snapshot_id
        db_session.add(row)
    db_session.flush()
    return snapshot


ASOF_AFTER_FILE = datetime(2026, 5, 30, 3, 0, tzinfo=timezone.utc)
ET = ZoneInfo("America/New_York")


class TestNasdaqTraderListingAdapter:
    def test_present_symbol_is_listed_active(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("aapl", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert resp.data.matched_symbol == "AAPL"
        assert resp.data.pit_knowable_at_asof is True

    def test_common_word_substrings_are_not_noncommon_descriptors(self):
        cases = [
            ("ACU", "Acme United Corporation. Common Stock"),
            ("UBCP", "United Bancorp, Inc. - Common Stock"),
            ("UG", "United-Guardian, Inc. - Common Stock"),
            ("UNITY", "Unity Bancorp, Inc. - Common Stock"),
            ("UNITI", "Uniti Group Inc. - Common Stock"),
            ("FUNDG", "Example Funding Corp. Common Stock"),
            ("FUNDX", "Example Fundamental Corp. Common Stock"),
            ("FNKO", "Funko, Inc. - Common Stock"),
            ("NOTEZ", "Example Noted Corp. Common Stock"),
            ("NOTEW", "Example Noteworthy Corp. Common Stock"),
            ("BONDH", "Example Bondholder Corp. Common Stock"),
            ("RGHTZ", "Example Righthand Corp. Common Stock"),
        ]
        for symbol, name in cases:
            adapter, _ = _adapter_for(
                _nasdaq_directory(_rows("N", 1001)),
                _other_directory([(symbol, name), *_rows("O", 1000)]),
                _adds_deletes(),
                _halt_rss(),
            )

            resp = adapter.get_listing_status(symbol, asof=ASOF_AFTER_FILE)

            assert resp.ok
            assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
            assert resp.data.matched_symbol == symbol

    def test_cvi_and_lp_equity_units_remain_eligible(self):
        # CVI is CVR Energy common stock, not a contingent-value right. These
        # LP listings are equity-like common/class-share rows, so future
        # descriptor vocabulary must not reject them as generic non-common units.
        cases = [
            ("CVI", "CVR Energy Inc. Common Stock"),
            ("PAGP", "Plains GP Holdings, L.P. Class A Shares"),
            ("HESM", "Hess Midstream LP Class A Share"),
            ("NRP", "Natural Resource Partners L.P. Common Units Representing Limited Partner Interests"),
        ]
        for symbol, name in cases:
            adapter, _ = _adapter_for(
                _nasdaq_directory(_rows("N", 1001)),
                _other_directory([(symbol, name), *_rows("O", 1000)]),
                _adds_deletes(),
                _halt_rss(),
            )

            resp = adapter.get_listing_status(symbol, asof=ASOF_AFTER_FILE)

            assert resp.ok
            assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
            assert resp.data.matched_symbol == symbol

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
            _nasdaq_directory([("CVGW", "Calavo Growers, Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes([
                ("CVGW", "Calavo Growers, Inc.", "Delete", "Delete", "Delete", "5/29/2026", "Q")
            ]),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("CVGW", asof=ASOF_AFTER_FILE)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trading_system_adds_deletes_delete"

    def test_reason_d_halt_is_delisted_but_ludp_is_not(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory([("DEAD", "Dead Co. - Common Stock"), *_rows("N", 1000)]),
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

    def test_future_knowledge_delete_blocks_directory_listed_active(self):
        asof = datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc)
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0529202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0529202618:15||||||"),
            _adds_deletes(
                [("AAPL", "Apple Inc.", "Delete", "Delete", "Delete", "5/29/2026", "Q")],
                footer="File Creation Time: 0529202618:15|||||",
            ),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("AAPL", asof=asof)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.reason == "adds_deletes_delete_not_knowable_at_asof"
        assert resp.data.pit_knowable_at_asof is False

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
            _adds_deletes(),
            _halt_rss(),
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
            _adds_deletes(),
            _halt_rss(),
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
            ("UNITZ", "Example Co. - Units", "N", "N"),
            ("WRTZ", "Example Co. - Warrants", "N", "N"),
            ("RGHTZ", "Example Co. - Rights", "N", "N"),
            ("PREFZ", "Example Co. - Preferred Stock", "N", "N"),
            ("NOTEZ", "Example Co. - Notes", "N", "N"),
            ("BONDZ", "Example Co. - Bonds", "N", "N"),
            ("DEBTZ", "Example Co. - Debenture", "N", "N"),
            ("FUNDZ", "Example Fund", "N", "N"),
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

    def test_same_day_live_snapshot_is_daily_grain_knowable_at_session_close(self):
        session_close_before_file_creation = datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc)
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("AAPL", asof=session_close_before_file_creation)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert resp.data.pit_knowable_at_asof is True

    def test_same_day_live_snapshot_before_session_close_is_not_knowable(self):
        pre_close = datetime(2026, 5, 29, 9, 45, tzinfo=ET)
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("AAPL", asof=pre_close)

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.reason == "directory_match_not_knowable_at_asof"
        assert resp.data.pit_knowable_at_asof is False

    def test_same_day_live_snapshot_uses_real_half_day_close(self):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 1127202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 1127202618:15||||||"),
            _adds_deletes(footer="File Creation Time: 1127202618:15|||||"),
            _halt_rss(channel_pub="Fri, 27 Nov 2026 22:15:00 GMT"),
        )

        at_half_day_close = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 11, 27, 13, 0, tzinfo=ET),
        )

        assert at_half_day_close.ok
        assert at_half_day_close.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert at_half_day_close.data.pit_knowable_at_asof is True

        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 1127202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 1127202618:15||||||"),
            _adds_deletes(footer="File Creation Time: 1127202618:15|||||"),
            _halt_rss(channel_pub="Fri, 27 Nov 2026 22:15:00 GMT"),
        )
        before_half_day_close = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 11, 27, 12, 59, tzinfo=ET),
        )

        assert before_half_day_close.ok
        assert before_half_day_close.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert before_half_day_close.data.reason == "directory_match_not_knowable_at_asof"

    def test_live_snapshot_future_day_stays_inconclusive_for_past_asof(self):
        past_asof = datetime(2026, 5, 28, 20, 0, tzinfo=timezone.utc)
        adapter, _ = _adapter_for(
            _nasdaq_directory([("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]),
            _other_directory(_rows("O", 1001)),
            _adds_deletes(),
            _halt_rss(),
        )

        resp = adapter.get_listing_status("AAPL", asof=past_asof)

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

    def test_halt_future_knowledge_does_not_hide_known_d(self):
        for items in (
            [
                ("DEAD", "D", "05/29/2026", "19:50:00.000", "Fri, 29 May 2026 23:59:00 GMT"),
                ("DEAD", "D", "05/29/2026", "19:51:00.000", "Fri, 29 May 2026 23:40:00 GMT"),
            ],
            [
                ("DEAD", "D", "05/29/2026", "19:51:00.000", "Fri, 29 May 2026 23:40:00 GMT"),
                ("DEAD", "D", "05/29/2026", "19:50:00.000", "Fri, 29 May 2026 23:59:00 GMT"),
            ],
        ):
            adapter, _ = _adapter_for(
                _nasdaq_directory(_rows("N", 1001)),
                _other_directory(_rows("O", 1001)),
                _adds_deletes(),
                _halt_rss(items),
            )

            resp = adapter.get_listing_status(
                "DEAD",
                asof=datetime(2026, 5, 29, 23, 55, tzinfo=timezone.utc),
            )

            assert resp.ok
            assert resp.data.status is NasdaqListingStatus.DELISTED
            assert resp.data.reason == "trade_halt_reason_D_security_deletion"
            assert resp.data.pit_knowable_at_asof is True

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
            _adds_deletes(),
            _halt_rss(),
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
            _halt_rss(),
        )

        captured = adapter.archive_current_snapshot(db_session, asof=ASOF_AFTER_FILE)

        assert captured.ok
        assert captured.data.inserted_snapshots == 4
        assert captured.data.inserted_rows == 2002
        assert HALT_RSS in captured.data.captured_sources

        before_capture = datetime(2026, 5, 28, 20, 0, tzinfo=timezone.utc)
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

    def test_archive_capture_persists_halt_rss_and_replays_reason_d(
        self,
        db_session,
    ):
        directory = _nasdaq_directory(
            [("HALTD", "Halt Deleted Co. - Common Stock"), *_rows("N", 1000)],
            footer="File Creation Time: 0601202618:15|||||||",
        )
        other = _other_directory(_rows("O", 1001), footer="File Creation Time: 0601202618:15||||||")
        adds = _adds_deletes(footer="File Creation Time: 0601202618:15|||||")
        halt_with_delete = _halt_rss(
            [("HALTD", "D", "06/01/2026", "15:50:00.000", "Mon, 01 Jun 2026 19:55:00 GMT")],
            channel_pub="Mon, 01 Jun 2026 20:00:00 GMT",
        )
        changed_halt_feed_without_haltd = _halt_rss(
            [("OTHERD", "D", "06/01/2026", "15:51:00.000", "Mon, 01 Jun 2026 19:56:00 GMT")],
            channel_pub="Mon, 01 Jun 2026 20:00:00 GMT",
        )
        adapter, _ = _adapter_for(
            directory,
            other,
            adds,
            halt_with_delete,
            directory,
            other,
            adds,
            halt_with_delete,
            directory,
            other,
            adds,
            changed_halt_feed_without_haltd,
        )

        archive_asof = datetime(2026, 6, 1, 18, 15, tzinfo=ET)
        first = adapter.archive_current_snapshot(db_session, asof=archive_asof)
        second = adapter.archive_current_snapshot(db_session, asof=archive_asof)
        third = adapter.archive_current_snapshot(db_session, asof=archive_asof)

        assert first.ok
        assert HALT_RSS in first.data.captured_sources
        assert first.data.inserted_snapshots == 4
        assert first.data.inserted_rows == 2003
        assert second.ok
        assert second.data.inserted_snapshots == 0
        assert second.data.existing_snapshots == 4
        assert second.data.inserted_rows == 0
        assert third.ok
        assert third.data.inserted_snapshots == 1
        assert third.data.inserted_rows == 1

        replay = adapter.get_listing_status(
            "HALTD",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert replay.ok
        assert replay.data.status is NasdaqListingStatus.DELISTED
        assert replay.data.reason == "trade_halt_reason_D_security_deletion"

    def test_archive_capture_degrades_when_halt_rss_fails(
        self,
        db_session,
    ):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(text=_nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)]
            )),
            _mock_response(text=_other_directory(_rows("O", 1001))),
            _mock_response(text=_adds_deletes()),
            requests.exceptions.Timeout(),
        ]
        adapter = NasdaqTraderListingAdapter(session=session)

        captured = adapter.archive_current_snapshot(db_session, asof=ASOF_AFTER_FILE)

        assert captured.ok
        assert captured.data.failed_sources == (HALT_RSS,)
        assert HALT_RSS not in captured.data.captured_sources
        assert captured.data.inserted_snapshots == 3
        assert captured.data.inserted_rows == 2002
        assert captured.lineage.data_quality_flags["failed_sources"] == [HALT_RSS]

    def test_archive_same_session_evening_snapshot_covers_session_close_asof(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [
                    ("AAPL", "Apple Inc. - Common Stock"),
                    ("AACBU", "Artius II Acquisition Inc. - Units"),
                    ("ACIU", "AC Immune SA - Common Stock"),
                    *_rows("N", 1000),
                ],
                footer="File Creation Time: 0601202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0601202618:15||||||"),
            _adds_deletes(footer="File Creation Time: 0601202618:15|||||"),
            _halt_rss(channel_pub="Mon, 01 Jun 2026 22:15:00 GMT"),
        )
        captured = adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 1, 18, 15, tzinfo=ET),
        )
        assert captured.ok

        session_close = datetime(2026, 6, 1, 16, 0, tzinfo=ET)
        pre_close = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 9, 45, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )
        archived = adapter.get_listing_status(
            "AAPL",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )
        after_source = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 20, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )
        next_day = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 2, 0, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )
        absent = adapter.get_listing_status(
            "ZZZZNOTREAL",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )
        unit = adapter.get_listing_status(
            "AACB.U",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )
        common_ending_u = adapter.get_listing_status(
            "ACI.U",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )

        assert pre_close.ok
        assert pre_close.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert pre_close.data.reason == "directory_match_not_knowable_at_asof"
        assert pre_close.data.pit_knowable_at_asof is False
        assert archived.ok
        assert archived.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert archived.data.pit_knowable_at_asof is True
        assert after_source.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert after_source.data.pit_knowable_at_asof is True
        assert next_day.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert next_day.data.reason == "archived_snapshot_stale_for_asof"
        assert absent.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert absent.data.reason == "symbol_absent_from_archived_directory"
        assert absent.data.pit_knowable_at_asof is False
        assert unit.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert unit.data.matched_symbol == "AACBU"
        assert common_ending_u.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert common_ending_u.data.reason == "directory_record_not_common_stock_listing"

    def test_archive_unions_per_source_snapshots_under_timestamp_skew(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0601202618:15|||||||",
            ),
            _other_directory(
                [("IBM", "International Business Machines Corporation Common Stock"), *_rows("O", 1000)],
                footer="File Creation Time: 0601202618:14||||||",
            ),
            _adds_deletes(footer="File Creation Time: 0601202618:15|||||"),
            _halt_rss(channel_pub="Mon, 01 Jun 2026 22:15:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 1, 18, 15, tzinfo=ET),
        ).ok

        session_close = datetime(2026, 6, 1, 16, 0, tzinfo=ET)
        ibm = adapter.get_listing_status(
            "IBM",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )

        assert ibm.ok
        assert ibm.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert ibm.data.matched_symbol == "IBM"
        assert ibm.data.pit_knowable_at_asof is True

    def test_archive_unions_per_source_snapshots_under_reverse_timestamp_skew(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0601202618:14|||||||",
            ),
            _other_directory(
                [("IBM", "International Business Machines Corporation Common Stock"), *_rows("O", 1000)],
                footer="File Creation Time: 0601202618:15||||||",
            ),
            _adds_deletes(footer="File Creation Time: 0601202618:15|||||"),
            _halt_rss(channel_pub="Mon, 01 Jun 2026 22:15:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 1, 18, 15, tzinfo=ET),
        ).ok

        session_close = datetime(2026, 6, 1, 16, 0, tzinfo=ET)
        aapl = adapter.get_listing_status(
            "AAPL",
            asof=session_close,
            archive_session=db_session,
            use_live=False,
        )

        assert aapl.ok
        assert aapl.data.status is NasdaqListingStatus.LISTED_ACTIVE
        assert aapl.data.matched_symbol == "AAPL"
        assert aapl.data.pit_knowable_at_asof is True

    def test_archive_same_day_delete_outranks_directory_presence(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0601202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0601202618:15||||||"),
            _adds_deletes(
                [("AAPL", "Apple Inc.", "Delete", "Delete", "Delete", "6/1/2026", "Q")],
                footer="File Creation Time: 0601202615:50|||||",
            ),
            _halt_rss(channel_pub="Mon, 01 Jun 2026 22:15:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 1, 18, 15, tzinfo=ET),
        ).ok

        resp = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trading_system_adds_deletes_delete"

    def test_archive_unions_all_same_day_delete_snapshots_before_directory(
        self,
        db_session,
    ):
        _insert_archive_snapshot(
            db_session,
            NASDAQ_LISTED,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=NASDAQ_LISTED,
                    symbol="DELA",
                    normalized_symbol="DELA",
                    security_name="Delete A Common Stock",
                    raw_json=json.dumps({
                        "ETF": "N",
                        "Security Name": "Delete A Common Stock",
                        "Symbol": "DELA",
                        "Test Issue": "N",
                    }),
                ),
            ],
        )
        _insert_archive_snapshot(
            db_session,
            ADDS_DELETES,
            datetime(2026, 6, 1, 15, 50, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=ADDS_DELETES,
                    symbol="DELA",
                    normalized_symbol="DELA",
                    security_name="Delete A",
                    action="Delete|Delete|Delete",
                    effective_date="2026-06-01",
                    raw_json=json.dumps({"Symbol": "DELA"}),
                )
            ],
        )
        _insert_archive_snapshot(
            db_session,
            ADDS_DELETES,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=ADDS_DELETES,
                    symbol="OTHER",
                    normalized_symbol="OTHER",
                    security_name="Other",
                    action="Add|Add|Add",
                    effective_date="2026-06-01",
                    raw_json=json.dumps({"Symbol": "OTHER"}),
                ),
            ],
        )
        adapter = NasdaqTraderListingAdapter()

        resp = adapter.get_listing_status(
            "DELA",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trading_system_adds_deletes_delete"

    def test_archive_known_delete_wins_over_future_knowledge_delete(
        self,
        db_session,
    ):
        _insert_archive_snapshot(
            db_session,
            NASDAQ_LISTED,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=NASDAQ_LISTED,
                    symbol="DELF",
                    normalized_symbol="DELF",
                    security_name="Delete Fold Common Stock",
                    raw_json=json.dumps({
                        "ETF": "N",
                        "Security Name": "Delete Fold Common Stock",
                        "Symbol": "DELF",
                        "Test Issue": "N",
                    }),
                )
            ],
        )
        _insert_archive_snapshot(
            db_session,
            ADDS_DELETES,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=ADDS_DELETES,
                    symbol="DELF",
                    normalized_symbol="DELF",
                    security_name="Delete Fold",
                    action="Delete|Delete|Delete",
                    effective_date="2026-06-01",
                    raw_json=json.dumps({"Symbol": "DELF"}),
                )
            ],
        )
        _insert_archive_snapshot(
            db_session,
            ADDS_DELETES,
            datetime(2026, 6, 1, 15, 50, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=ADDS_DELETES,
                    symbol="DELF",
                    normalized_symbol="DELF",
                    security_name="Delete Fold",
                    action="Delete|Delete|Delete",
                    effective_date="2026-06-01",
                    raw_json=json.dumps({"Symbol": "DELF"}),
                )
            ],
        )
        adapter = NasdaqTraderListingAdapter()

        resp = adapter.get_listing_status(
            "DELF",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trading_system_adds_deletes_delete"

    def test_archive_future_knowledge_delete_blocks_directory_presence(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0601202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0601202618:15||||||"),
            _adds_deletes(
                [("AAPL", "Apple Inc.", "Delete", "Delete", "Delete", "6/1/2026", "Q")],
                footer="File Creation Time: 0601202618:15|||||",
            ),
            _halt_rss(channel_pub="Mon, 01 Jun 2026 22:15:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 1, 18, 15, tzinfo=ET),
        ).ok

        resp = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.reason == "adds_deletes_delete_not_knowable_at_asof"

    def test_archive_same_day_halt_reason_d_outranks_directory_presence(
        self,
        db_session,
    ):
        _insert_archive_snapshot(
            db_session,
            NASDAQ_LISTED,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=NASDAQ_LISTED,
                    symbol="AAPL",
                    normalized_symbol="AAPL",
                    security_name="Apple Inc. - Common Stock",
                    raw_json=json.dumps({
                        "ETF": "N",
                        "Security Name": "Apple Inc. - Common Stock",
                        "Symbol": "AAPL",
                        "Test Issue": "N",
                    }),
                )
            ],
        )
        _insert_archive_snapshot(
            db_session,
            HALT_RSS,
            datetime(2026, 6, 1, 15, 55, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=HALT_RSS,
                    symbol="AAPL",
                    normalized_symbol="AAPL",
                    security_name="Apple Inc.",
                    reason_code="D",
                    effective_date="2026-06-01",
                    raw_json=json.dumps({
                        "IssueSymbol": "AAPL",
                        "ReasonCode": "D",
                        "pubDate": "Mon, 01 Jun 2026 19:55:00 GMT",
                    }),
                )
            ],
        )

        adapter = NasdaqTraderListingAdapter()
        resp = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.DELISTED
        assert resp.data.reason == "trade_halt_reason_D_security_deletion"

    def test_archive_directory_raw_gaps_fail_closed(
        self,
        db_session,
    ):
        _insert_archive_snapshot(
            db_session,
            NASDAQ_LISTED,
            datetime(2026, 6, 1, 18, 15, tzinfo=ET),
            [
                NasdaqListingSnapshotRow(
                    source_type=NASDAQ_LISTED,
                    symbol="TESTY",
                    normalized_symbol="TESTY",
                    security_name="Test Issue Common Stock",
                    raw_json=json.dumps({
                        "ETF": "N",
                        "Security Name": "Test Issue Common Stock",
                        "Symbol": "TESTY",
                    }),
                ),
                NasdaqListingSnapshotRow(
                    source_type=NASDAQ_LISTED,
                    symbol="BADRAW",
                    normalized_symbol="BADRAW",
                    security_name="Bad Raw Common Stock",
                    raw_json=json.dumps(["not", "a", "dict"]),
                ),
            ],
        )

        adapter = NasdaqTraderListingAdapter()
        for symbol in ("TESTY", "BADRAW"):
            resp = adapter.get_listing_status(
                symbol,
                asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
                archive_session=db_session,
                use_live=False,
            )

            assert resp.ok
            assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
            assert resp.data.reason == "directory_record_not_common_stock_listing"

    def test_archive_prior_session_snapshot_does_not_cover_later_asof(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0529202618:15|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0529202618:15||||||"),
            _adds_deletes(footer="File Creation Time: 0529202618:15|||||"),
            _halt_rss(channel_pub="Fri, 29 May 2026 22:15:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 5, 29, 18, 15, tzinfo=ET),
        ).ok

        resp = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.reason == "archived_snapshot_stale_for_asof"

    def test_archive_future_session_snapshot_does_not_answer_prior_asof(
        self,
        db_session,
    ):
        adapter, _ = _adapter_for(
            _nasdaq_directory(
                [("AAPL", "Apple Inc. - Common Stock"), *_rows("N", 1000)],
                footer="File Creation Time: 0602202600:00|||||||",
            ),
            _other_directory(_rows("O", 1001), footer="File Creation Time: 0602202600:00||||||"),
            _adds_deletes(footer="File Creation Time: 0602202600:00|||||"),
            _halt_rss(channel_pub="Tue, 02 Jun 2026 04:00:00 GMT"),
        )
        assert adapter.archive_current_snapshot(
            db_session,
            asof=datetime(2026, 6, 2, 0, 0, tzinfo=ET),
        ).ok

        resp = adapter.get_listing_status(
            "AAPL",
            asof=datetime(2026, 6, 1, 16, 0, tzinfo=ET),
            archive_session=db_session,
            use_live=False,
        )

        assert resp.ok
        assert resp.data.status is NasdaqListingStatus.INCONCLUSIVE
        assert resp.data.reason == "no_archived_snapshot_for_asof"

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
        asof=datetime(2026, 5, 28, 3, 0, tzinfo=timezone.utc),
        halt_date=date(2026, 5, 27),
    )
    assert halt_resp.ok
    reason_d = _status_from_halt_events(
        "CVGW",
        datetime(2026, 5, 28, 3, 0, tzinfo=timezone.utc),
        halt_resp.data,
    )
    assert reason_d is not None
    assert reason_d.status is NasdaqListingStatus.DELISTED
