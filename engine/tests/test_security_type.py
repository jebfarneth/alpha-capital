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
from alpha.db.models import (
    Base,
    CanonicalUniverseScan,
    SecurityProfile,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.runner import run_job
from alpha.jobs.security_type import (
    ADR,
    CLOSED_END_FUND,
    COMMON_STOCK,
    ETF,
    MUTUAL_FUND,
    BUSINESS_DEVELOPMENT_COMPANY,
    NON_COMMON_TYPES,
    PREFERRED,
    RIGHT,
    SPAC_OR_BLANK_CHECK,
    UNIT,
    UNKNOWN,
    WARRANT,
    CLASSIFIER_VERSION,
    REFRESH_STATUS_ENRICHED,
    REFRESH_STATUS_FAILED,
    REFRESH_STATUS_NO_DATA,
    REFRESH_STATUS_RETRYABLE_ERROR,
    SecurityTypeEnrichmentJob,
    classify_security_type,
)
from alpha.jobs.universe_builder import UniverseBuilderJob, _security_profile_cache_hash


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

    def test_mutual_funds_plural_from_name(self):
        st, _ = classify_security_type(_profile(company_name="Gabelli Funds Inc"))
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

    def test_spac_from_acquisition_corporation(self):
        st, reason = classify_security_type(
            _profile(company_name="Black Hawk Acquisition Corporation")
        )
        assert st == SPAC_OR_BLANK_CHECK
        assert reason == "name_contains:ACQUISITION CORPORATION"

    def test_spac_from_blank_check(self):
        st, _ = classify_security_type(_profile(company_name="Blank Check Corp"))
        assert st == SPAC_OR_BLANK_CHECK

    def test_spac_from_shell_company_industry(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Perceptive Capital Solutions Corp",
                sector="Financial Services",
                industry="Shell Companies",
            )
        )
        assert st == SPAC_OR_BLANK_CHECK
        assert reason == "industry:SHELL_COMPANIES"

    @pytest.mark.parametrize(
        "name",
        [
            "Chenghe Acquisition III Co. Class A Ordinary Share",
            "YHN Acquisition I Limited",
            "Newbridge Acquisition Limited Class A Ordinary Share",
            "Black Spade Acquisition III Co",
        ],
    )
    def test_spac_from_acquisition_sequence_names(self, name):
        st, reason = classify_security_type(_profile(company_name=name))
        assert st == SPAC_OR_BLANK_CHECK
        assert reason == "name_pattern:ACQUISITION_SEQUENCE"

    def test_spac_from_investment_corp_sequence_name(self):
        st, reason = classify_security_type(_profile(company_name="Origin Investment Corp I"))
        assert st == SPAC_OR_BLANK_CHECK
        assert reason == "name_pattern:INVESTMENT_CORP_SEQUENCE"

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

    def test_trust_asset_manager_not_auto_closed_end_fund(self):
        st, _ = classify_security_type(
            _profile(company_name="First Trust Inc", industry="Asset Management")
        )
        assert st == COMMON_STOCK

    def test_financial_acquisition_business_not_auto_spac(self):
        st, _ = classify_security_type(
            _profile(
                company_name="Talent Acquisition Solutions Inc",
                sector="Financial Services",
            )
        )
        assert st == COMMON_STOCK

    def test_bdc_from_business_development_name(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Example Business Development Company",
                sector="Financial Services",
                industry="Asset Management",
            )
        )
        assert st == BUSINESS_DEVELOPMENT_COMPANY
        assert reason == "name_contains:BUSINESS_DEVELOPMENT_COMPANY"

    def test_bdc_from_capital_corporation_financial_profile(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Ares Capital Corporation",
                sector="Financial Services",
                industry="Asset Management",
            )
        )
        assert st == BUSINESS_DEVELOPMENT_COMPANY
        assert reason == "name_industry:CAPITAL_CORPORATION_BDC"

    def test_bdc_from_raw_description(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Hercules Capital, Inc.",
                sector="Financial Services",
                industry="Asset Management",
            ),
            raw_json={"description": "Hercules Capital, Inc. is a business development company."},
        )
        assert st == BUSINESS_DEVELOPMENT_COMPANY
        assert reason == "raw_description:BUSINESS_DEVELOPMENT_COMPANY"

    def test_capital_corporation_outside_financial_profile_not_bdc(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Acme Capital Corporation",
                sector="Technology",
                industry="Software",
            )
        )
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

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

    def test_non_us_issuer_on_us_exchange_is_not_auto_adr(self):
        st, reason = classify_security_type(
            _profile(company_name="Banco Santander SA", country="ES", exchange="NYSE")
        )
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

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
        assert BUSINESS_DEVELOPMENT_COMPANY in NON_COMMON_TYPES
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


def _mock_retryable_response():
    return AdapterResponse(
        data=None,
        lineage=_lineage(),
        error=ProviderError(
            provider="FMP",
            endpoint="/stable/profile",
            status_code=429,
            error_type="rate_limit",
            message="Rate limited",
            retryable=True,
        ),
    )


def _mock_auth_error_response():
    return AdapterResponse(
        data=None,
        lineage=_lineage(),
        error=ProviderError(
            provider="FMP",
            endpoint="/stable/profile",
            status_code=403,
            error_type="auth",
            message="Auth failed",
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
        assert prof.refresh_status == REFRESH_STATUS_ENRICHED
        assert prof.source_provider == "FMP"
        assert prof.classifier_version == CLASSIFIER_VERSION
        assert prof.classification_input_hash
        assert prof.classification_output_hash

    def test_updates_existing_profile(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ACME", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_NO_DATA,
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
        assert prof.refresh_status == REFRESH_STATUS_ENRICHED

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
        assert prof.refresh_status == REFRESH_STATUS_NO_DATA
        assert prof.classifier_version == CLASSIFIER_VERSION
        assert prof.classification_output_hash

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
        assert prof.refresh_status == REFRESH_STATUS_FAILED
        assert prof.classification_reason == "profile_fetch_exception:RuntimeError"

    def test_retryable_error_retries_then_records_retryable_status(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.side_effect = [
            _mock_retryable_response(),
            _mock_retryable_response(),
        ]
        sleeps = []

        job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=adapter,
            symbols=["SLOW"],
            max_retries=1,
            retry_backoff_seconds=0.25,
            sleep_fn=sleeps.append,
        )
        result = run_job(db_session, job)

        assert result.ok
        assert adapter.get_company_profile.call_count == 2
        assert sleeps == [0.25]
        assert result.metrics["retryable_error_count"] == 1
        assert result.metrics["no_data_count"] == 0
        assert result.metrics["retry_attempt_count"] == 1

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "SLOW"
        ).one()
        assert prof.security_type == UNKNOWN
        assert prof.refresh_status == REFRESH_STATUS_RETRYABLE_ERROR
        assert prof.classification_reason == "profile_fetch_retryable:rate_limit"

    def test_retryable_error_then_success_does_not_mark_no_data(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.side_effect = [
            _mock_retryable_response(),
            _mock_profile_response(_profile("SLOW")),
        ]
        sleeps = []

        job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=adapter,
            symbols=["SLOW"],
            max_retries=1,
            retry_backoff_seconds=0.25,
            sleep_fn=sleeps.append,
        )
        result = run_job(db_session, job)

        assert result.metrics["enriched_count"] == 1
        assert result.metrics["retryable_error_count"] == 0
        assert result.metrics["no_data_count"] == 0
        assert result.metrics["retry_attempt_count"] == 1
        assert sleeps == [0.25]

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "SLOW"
        ).one()
        assert prof.security_type == COMMON_STOCK
        assert prof.refresh_status == REFRESH_STATUS_ENRICHED

    def test_non_retryable_error_is_failed_not_no_data(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_auth_error_response()

        job = SecurityTypeEnrichmentJob(
            session=db_session, adapter=adapter, symbols=["AUTH"],
        )
        result = run_job(db_session, job)

        assert result.metrics["failed_count"] == 1
        assert result.metrics["no_data_count"] == 0
        assert result.metrics["retryable_error_count"] == 0

        prof = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "AUTH"
        ).one()
        assert prof.refresh_status == REFRESH_STATUS_FAILED
        assert prof.classification_reason == "profile_fetch_failed:auth"

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
        assert prof.classification_input_hash == stable_hash({
            "symbol": "RAWP",
            "company_name": "Raw Payload Corp",
            "sector": "Technology",
            "industry": "Software",
            "exchange": "NASDAQ",
            "country": "US",
            "is_etf": False,
            "raw": {"isFund": False},
        })
        assert prof.classification_output_hash == stable_hash({
            "classifier_version": CLASSIFIER_VERSION,
            "security_type": COMMON_STOCK,
            "classification_reason": "profile_fields_present",
            "refresh_status": REFRESH_STATUS_ENRICHED,
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

    def test_derived_symbols_include_suffix_excluded_candidates(self, db_session):
        db_session.add_all([
            UniverseSnapshot(
                ticker="INCL",
                asof_timestamp=_ts(),
                operating_universe_inclusion=True,
                exclusion_reason=None,
            ),
            UniverseSnapshot(
                ticker="ABCDX",
                asof_timestamp=_ts(),
                operating_universe_inclusion=False,
                exclusion_reason="non_common_symbol_suffix",
            ),
            UniverseSnapshot(
                ticker="PENY",
                asof_timestamp=_ts(),
                operating_universe_inclusion=False,
                exclusion_reason="price_below_3",
            ),
        ])
        db_session.flush()

        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.return_value = _mock_no_data_response()

        job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=adapter,
            retry_backoff_seconds=0,
        )
        result = run_job(db_session, job)

        called_symbols = {
            call.args[0] for call in adapter.get_company_profile.call_args_list
        }
        assert result.ok
        assert called_symbols == {"INCL", "ABCDX"}

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
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
            classification_output_hash="common-hash",
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
        assert result.metrics["security_profile_required_count"] == 1
        assert result.metrics["security_profile_enriched_count"] == 1
        assert result.metrics["security_profile_coverage_ratio"] == 1.0
        assert result.input_hashes["security_profile_cache"]
        scan = db_session.query(UniverseScan).one()
        assert scan.security_profile_cache_hash == result.input_hashes["security_profile_cache"]

    def test_non_common_type_excludes_with_reason(self, db_session):
        db_session.add(SecurityProfile(
            symbol="BADF", security_type=MUTUAL_FUND,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("BADF"), _stock("GOOD")],
            lineage=_mock_lineage_screener(),
        )
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
        assert result.metrics["security_profile_cache_miss_included_count"] == 1
        assert result.metrics["security_profile_cache_miss_required_count"] == 1
        assert result.metrics["security_profile_coverage_ratio"] == 0.0

    def test_required_security_profile_coverage_blocks_canonical_on_miss(self, db_session):
        resp = AdapterResponse(data=[_stock("NOCACHE")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(
            session=db_session,
            screener_response=resp,
            require_security_profile_cache=True,
            min_security_profile_coverage=1.0,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert not result.ok
        assert result.status == "failed"
        assert result.metrics["failure_stage"] == "security_profile_coverage"
        assert result.metrics["security_profile_required_count"] == 1
        assert result.metrics["security_profile_enriched_count"] == 0
        assert result.metrics["security_profile_coverage_ratio"] == 0.0
        assert result.metrics["security_profile_cache_miss_required_count"] == 1

        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "failed"
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_required_security_profile_coverage_allows_complete_cache(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ACME", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ACME")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(
            session=db_session,
            screener_response=resp,
            require_security_profile_cache=True,
            min_security_profile_coverage=1.0,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["security_profile_coverage_ratio"] == 1.0
        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "finished"
        assert db_session.query(CanonicalUniverseScan).count() == 1

    def test_unknown_type_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="MYSTK", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_NO_DATA,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("MYSTK"), _stock("GOOD")],
            lineage=_mock_lineage_screener(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "MYSTK"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_profile_unresolved:no_data"
        assert snap.security_type == UNKNOWN
        assert result.metrics["security_type_unknown_count"] == 1
        assert result.metrics["security_profile_unresolved_count"] == 1

    def test_retryable_profile_status_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="SLOW", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_RETRYABLE_ERROR,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("SLOW")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "SLOW"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_profile_unresolved:retryable_error"
        assert result.metrics["security_profile_unresolved_count"] == 1

    def test_stale_profile_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="OLD", security_type=COMMON_STOCK,
            last_refreshed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("OLD")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(
            session=db_session,
            screener_response=resp,
            profile_cache_max_age_days=7,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "OLD"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_profile_stale"
        assert result.metrics["security_profile_stale_count"] == 1

    def test_stale_non_common_profile_does_not_override_suffix_reason(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ABCDX", security_type=MUTUAL_FUND,
            last_refreshed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ABCDX")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(
            session=db_session,
            screener_response=resp,
            profile_cache_max_age_days=7,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "ABCDX"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "non_common_symbol_suffix"
        assert snap.security_type == MUTUAL_FUND
        assert result.metrics["security_profile_stale_count"] == 1
        assert "mutual_fund" not in result.metrics["security_type_exclusion_counts"]

    def test_common_stock_cache_rescues_suffix_fallback(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ABCDX", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
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
        assert result.metrics["security_profile_coverage_ratio"] == 1.0

    def test_common_stock_cache_does_not_rescue_warrant_suffix(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ASTSW", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ASTSW")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "ASTSW"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "non_common_symbol_suffix"
        assert snap.security_type == COMMON_STOCK
        assert result.metrics["security_type_suffix_rescue_count"] == 0

    def test_non_common_cache_overrides_suffix_reason(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ABCDX", security_type=MUTUAL_FUND,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("ABCDX")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "ABCDX"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_type:mutual_fund"
        assert result.metrics["security_type_exclusion_counts"]["mutual_fund"] == 1

    def test_all_non_common_types_excluded(self, db_session):
        """Every canonical non-common type produces an exclusion."""
        for i, st in enumerate(sorted(NON_COMMON_TYPES)):
            sym = f"T{i:03d}"
            db_session.add(SecurityProfile(
                symbol=sym, security_type=st,
                last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            ))
        db_session.flush()

        stocks = [_stock(f"T{i:03d}") for i in range(len(NON_COMMON_TYPES))]
        stocks.append(_stock("GOOD"))
        resp = AdapterResponse(data=stocks, lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["included"] == 1
        assert result.metrics["excluded"] == len(NON_COMMON_TYPES)
        for st in NON_COMMON_TYPES:
            assert f"security_type:{st}" in result.metrics["exclusion_counts"]

    def test_hard_filter_runs_before_security_type(self, db_session):
        """If hard filter already excludes, security_type doesn't override the reason."""
        db_session.add(SecurityProfile(
            symbol="PENY", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
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
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            classification_reason="test",
            classifier_version=CLASSIFIER_VERSION,
            classification_input_hash="input",
            classification_output_hash="output",
        ))
        db_session.flush()

        row = db_session.query(SecurityProfile).filter(
            SecurityProfile.symbol == "TEST"
        ).one()
        assert row.security_type == COMMON_STOCK
        assert row.classification_reason == "test"
        assert row.classifier_version == CLASSIFIER_VERSION
        assert row.classification_input_hash == "input"
        assert row.classification_output_hash == "output"

    def test_cache_hash_ignores_refresh_timestamp_inside_fresh_window(self):
        profile_a = SecurityProfile(
            symbol="ACME", security_type=COMMON_STOCK,
            refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
            classification_input_hash="input",
            classification_output_hash="output",
            last_refreshed_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
        )
        profile_b = SecurityProfile(
            symbol="ACME", security_type=COMMON_STOCK,
            refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
            classification_input_hash="input",
            classification_output_hash="output",
            last_refreshed_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        )

        hash_a = _security_profile_cache_hash(
            {"ACME": profile_a},
            ["ACME"],
            asof=_ts(),
            max_age_days=7,
        )
        hash_b = _security_profile_cache_hash(
            {"ACME": profile_b},
            ["ACME"],
            asof=_ts(),
            max_age_days=7,
        )

        assert hash_a == hash_b
