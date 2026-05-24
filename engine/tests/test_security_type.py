"""
Security-type enrichment and classifier tests.

  - Classifier rules for all canonical security_type values.
  - Enrichment job upserts profiles, handles errors per symbol.
  - Universe builder cache integration.
  - No live FMP calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
)
from alpha.data.fmp import FmpAdapter, FmpCompanyProfile, FmpScreenerResult
from alpha.db.models import Base, SecurityProfile, UniverseSnapshot
from alpha.jobs.runner import run_job
from alpha.jobs.security_type import (
    ADR,
    CLOSED_END_FUND,
    COMMON_STOCK,
    ETF,
    MUTUAL_FUND,
    NON_COMMON_TYPES,
    PREFERRED,
    RIGHT,
    SPAC_OR_BLANK_CHECK,
    UNIT,
    UNKNOWN,
    WARRANT,
    SecurityTypeEnrichmentJob,
    classify_security_type,
)
from alpha.jobs.universe_builder import UniverseBuilderJob


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


def _profile(
    symbol="ACME",
    company_name="Acme Corp",
    sector="Technology",
    industry="Software",
    exchange="NASDAQ",
    country="US",
    is_etf=False,
    is_actively_trading=True,
    ipo_date=None,
    raw=None,
) -> FmpCompanyProfile:
    return FmpCompanyProfile(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        industry=industry,
        exchange=exchange,
        country=country,
        is_etf=is_etf,
        is_actively_trading=is_actively_trading,
        ipo_date=ipo_date,
        raw=raw,
    )


def _lineage():
    return LineageMeta(
        provider="FMP",
        endpoint="/stable/profile",
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash="profile-hash",
        source_authority="FMP_Ultimate",
    )


# ===================================================================
# Classifier tests
# ===================================================================


class TestClassifier:
    def test_common_stock(self):
        st, reason = classify_security_type(_profile())
        assert st == COMMON_STOCK
        assert "profile_fields_present" in reason

    def test_etf_from_is_etf_flag(self):
        st, reason = classify_security_type(_profile(is_etf=True))
        assert st == ETF
        assert "is_etf=True" in reason

    def test_etf_from_sector(self):
        st, _ = classify_security_type(_profile(sector="ETF", is_etf=False))
        assert st == ETF

    def test_mutual_fund_from_name(self):
        st, _ = classify_security_type(_profile(company_name="Vanguard Growth Fund"))
        assert st == MUTUAL_FUND

    def test_closed_end_fund_from_name(self):
        st, _ = classify_security_type(_profile(company_name="Nuveen Closed-End Fund"))
        assert st == CLOSED_END_FUND

    def test_closed_end_fund_from_fund_plus_closed(self):
        st, _ = classify_security_type(_profile(company_name="XYZ Income Fund Closed"))
        assert st == CLOSED_END_FUND

    def test_adr_from_name(self):
        st, _ = classify_security_type(_profile(company_name="Alibaba ADR"))
        assert st == ADR

    def test_adr_from_american_depositary(self):
        st, _ = classify_security_type(_profile(company_name="TSMC American Depositary Shares"))
        assert st == ADR

    def test_preferred_from_name(self):
        st, _ = classify_security_type(_profile(company_name="XYZ Corp Preferred Series A"))
        assert st == PREFERRED

    def test_preferred_from_coupon_pattern(self):
        st, _ = classify_security_type(_profile(company_name="ABC 7.5% Fixed Rate Series B"))
        assert st == PREFERRED

    def test_warrant_from_name(self):
        st, _ = classify_security_type(_profile(company_name="SPAC Holdings Warrants"))
        assert st == WARRANT

    def test_unit_from_name(self):
        st, _ = classify_security_type(_profile(company_name="Newco Acquisition Units"))
        assert st == UNIT

    def test_unit_from_composite_name(self):
        st, _ = classify_security_type(_profile(company_name="XYZ Unit Consisting of Common"))
        assert st == UNIT

    def test_right_from_name(self):
        st, _ = classify_security_type(_profile(company_name="XYZ Subscription Right"))
        assert st == RIGHT

    def test_spac_from_acquisition_corp(self):
        st, _ = classify_security_type(_profile(company_name="Thunder Acquisition Corp"))
        assert st == SPAC_OR_BLANK_CHECK

    def test_spac_from_blank_check(self):
        st, _ = classify_security_type(_profile(company_name="Blank Check Corp"))
        assert st == SPAC_OR_BLANK_CHECK

    def test_unknown_when_no_data(self):
        st, reason = classify_security_type(_profile(company_name="", exchange=None))
        assert st == UNKNOWN
        assert "insufficient" in reason

    def test_common_stock_not_fooled_by_partial_keywords(self):
        """A company named 'Fundata Inc' should not be classified as fund."""
        st, _ = classify_security_type(_profile(company_name="Fundata Analytics Inc"))
        assert st == COMMON_STOCK

    def test_common_stock_not_fooled_by_spac_inside_space(self):
        st, _ = classify_security_type(_profile(company_name="AST SpaceMobile Inc"))
        assert st == COMMON_STOCK

    def test_common_stock_not_fooled_by_adr_inside_word(self):
        st, _ = classify_security_type(_profile(company_name="Adroit Infotech Inc"))
        assert st == COMMON_STOCK

    def test_common_stock_not_fooled_by_fund_inside_funding(self):
        st, _ = classify_security_type(_profile(company_name="Acme Funding Corp"))
        assert st == COMMON_STOCK

    def test_raw_provider_is_fund_flag_wins(self):
        st, reason = classify_security_type(
            _profile(company_name="Opaque Profile Inc"),
            raw_json={"isFund": True},
        )
        assert st == MUTUAL_FUND
        assert reason == "raw_flag:isFund"

    def test_raw_provider_is_adr_flag_wins(self):
        st, reason = classify_security_type(
            _profile(company_name="Foreign Issuer Inc"),
            raw_json={"isAdr": 1},
        )
        assert st == ADR
        assert reason == "raw_flag:isAdr"

    def test_non_us_issuer_on_us_exchange_is_not_common_stock(self):
        st, reason = classify_security_type(
            _profile(company_name="Banco Santander SA", country="ES", exchange="NYSE")
        )
        assert st == ADR
        assert reason == "profile_country:ES"

    def test_all_non_common_types_in_set(self):
        """Verify NON_COMMON_TYPES covers all non-common classifications."""
        assert ETF in NON_COMMON_TYPES
        assert MUTUAL_FUND in NON_COMMON_TYPES
        assert CLOSED_END_FUND in NON_COMMON_TYPES
        assert ADR in NON_COMMON_TYPES
        assert PREFERRED in NON_COMMON_TYPES
        assert WARRANT in NON_COMMON_TYPES
        assert UNIT in NON_COMMON_TYPES
        assert RIGHT in NON_COMMON_TYPES
        assert SPAC_OR_BLANK_CHECK in NON_COMMON_TYPES
        assert COMMON_STOCK not in NON_COMMON_TYPES
        assert UNKNOWN not in NON_COMMON_TYPES


# ===================================================================
# Enrichment job tests
# ===================================================================


def _mock_profile_response(profile: FmpCompanyProfile):
    return AdapterResponse(
        data=profile,
        lineage=_lineage(),
    )


def _mock_no_data_response():
    return AdapterResponse(
        data=None,
        lineage=_lineage(),
        error=ProviderError(
            provider="FMP",
            endpoint="/stable/profile",
            status_code=200,
            error_type="no_data",
            message="No profile found",
            retryable=False,
        ),
    )


class TestEnrichmentJob:
    def test_upserts_profile(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_profile_response(
            _profile("ACME", company_name="Acme Corp")
        )

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["ACME"],
        )
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["enriched_count"] == 1

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "ACME"
        ).one()
        assert prof.security_type == COMMON_STOCK
        assert prof.refresh_status == "enriched"
        assert prof.source_provider == "FMP"

    def test_updates_existing_profile(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ACME", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status="no_data",
        ))
        db_session.flush()

        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_profile_response(
            _profile("ACME", company_name="Acme Corp")
        )

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["ACME"],
        )
        run_job(db_session, job)

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "ACME"
        ).one()
        assert prof.security_type == COMMON_STOCK
        assert prof.refresh_status == "enriched"

    def test_handles_no_data_without_failing(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_no_data_response()

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["GHOST"],
        )
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["no_data_count"] == 1
        assert result.metrics["enriched_count"] == 0

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "GHOST"
        ).one()
        assert prof.security_type == UNKNOWN
        assert prof.refresh_status == "no_data"

    def test_handles_exception_without_failing(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.side_effect = RuntimeError("network error")

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["CRASH"],
        )
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["failed_count"] == 1

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "CRASH"
        ).one()
        assert prof.security_type == UNKNOWN
        assert prof.refresh_status == "failed"
        assert prof.classification_reason == "profile_fetch_exception:RuntimeError"

    def test_raw_profile_payload_is_persisted(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_profile_response(
            _profile(
                "RAWP",
                company_name="Raw Payload Corp",
                raw={"symbol": "RAWP", "companyName": "Raw Payload Corp", "isFund": False},
            )
        )

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["RAWP"],
        )
        run_job(db_session, job)

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "RAWP"
        ).one()
        assert json.loads(prof.raw_profile_json)["companyName"] == "Raw Payload Corp"
        assert prof.profile_payload_hash == stable_hash({
            "symbol": "RAWP",
            "companyName": "Raw Payload Corp",
            "isFund": False,
        })

    def test_multiple_symbols(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)

        def profile_for(ticker):
            if ticker == "ACME":
                return _mock_profile_response(_profile("ACME"))
            if ticker == "BADF":
                return _mock_profile_response(_profile("BADF", company_name="Bad Fund", is_etf=True))
            return _mock_no_data_response()

        adapter.get_company_profile.side_effect = profile_for

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter,
            symbols=["ACME", "BADF", "GONE"],
        )
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["enriched_count"] == 2
        assert result.metrics["no_data_count"] == 1
        counts = result.metrics["security_type_counts"]
        assert counts[COMMON_STOCK] == 1
        assert counts[ETF] == 1

    def test_security_type_counts_in_metrics(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_profile_response(
            _profile("SPAC", company_name="Thunder Acquisition Corp")
        )

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["SPAC"],
        )
        result = run_job(db_session, job)

        assert result.metrics["security_type_counts"][SPAC_OR_BLANK_CHECK] == 1


# ===================================================================
# Universe builder cache integration tests
# ===================================================================


def _mock_lineage_screener():
    return LineageMeta(
        provider="FMP",
        endpoint="/stable/company-screener",
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash="screener-hash",
        source_authority="mock",
    )


def _stock(symbol="ACME", **kw):
    defaults = dict(
        company_name=f"{symbol} Corp", market_cap=75_000_000, price=5.0,
        exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True,
    )
    defaults.update(kw)
    return FmpScreenerResult(symbol=symbol, **defaults)


class TestBuilderCacheIntegration:
    def test_cache_hit_populates_security_type(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ACME", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status="enriched",
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ACME")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "ACME"
        ).one()
        assert snap.security_type == COMMON_STOCK
        assert snap.operating_universe_inclusion is True
        assert result.metrics["security_profile_cache_hit_count"] == 1

    def test_non_common_type_excludes_with_reason(self, db_session):
        db_session.add(SecurityProfile(
            symbol="BADF", security_type=MUTUAL_FUND,
            last_refreshed_at=_ts(), refresh_status="enriched",
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("BADF")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "BADF"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_type:mutual_fund"
        assert snap.security_type == MUTUAL_FUND
        assert result.metrics["security_type_exclusion_counts"]["mutual_fund"] == 1

    def test_missing_cache_falls_back_without_crash(self, db_session):
        resp = AdapterResponse(data=[_stock("NOCACHE")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "NOCACHE"
        ).one()
        assert snap.operating_universe_inclusion is True
        assert snap.security_type is None
        assert result.metrics["security_profile_cache_miss_count"] == 1

    def test_unknown_type_does_not_exclude(self, db_session):
        db_session.add(SecurityProfile(
            symbol="MYSTK", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status="no_data",
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("MYSTK")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "MYSTK"
        ).one()
        assert snap.operating_universe_inclusion is True
        assert snap.security_type == UNKNOWN
        assert result.metrics["security_type_unknown_count"] == 1

    def test_common_stock_cache_rescues_suffix_fallback(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ABCDX", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status="enriched",
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ABCDX")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "ABCDX"
        ).one()
        assert snap.operating_universe_inclusion is True
        assert snap.exclusion_reason is None
        assert snap.security_type == COMMON_STOCK
        assert result.metrics["security_type_suffix_rescue_count"] == 1

    def test_all_non_common_types_excluded(self, db_session):
        """Every canonical non-common type produces an exclusion."""
        for i, st in enumerate(sorted(NON_COMMON_TYPES)):
            sym = f"T{i:03d}"
            db_session.add(SecurityProfile(
                symbol=sym, security_type=st,
                last_refreshed_at=_ts(), refresh_status="enriched",
            ))
        db_session.flush()

        stocks = [_stock(f"T{i:03d}") for i in range(len(NON_COMMON_TYPES))]
        resp = AdapterResponse(data=stocks, lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["included"] == 0
        assert result.metrics["excluded"] == len(NON_COMMON_TYPES)
        for st in NON_COMMON_TYPES:
            assert f"security_type:{st}" in result.metrics["exclusion_counts"]

    def test_hard_filter_runs_before_security_type(self, db_session):
        """If hard filter already excludes, security_type doesn't override the reason."""
        db_session.add(SecurityProfile(
            symbol="PENY", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status="enriched",
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("PENY", price=1.0)],  # fails price_below_3
            lineage=_mock_lineage_screener(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "PENY"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "price_below_3"


# ===================================================================
# Schema existence test
# ===================================================================

class TestSecurityProfileSchema:
    def test_table_exists(self, db_session):
        assert "security_profiles" in Base.metadata.tables

    def test_insert_and_query(self, db_session):
        db_session.add(SecurityProfile(
            symbol="TEST", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status="enriched",
            classification_reason="test",
        ))
        db_session.flush()

        row = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "TEST"
        ).one()
        assert row.security_type == COMMON_STOCK
        assert row.classification_reason == "test"
