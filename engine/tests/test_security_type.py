"""
Security-type enrichment and classifier tests.

  - Classifier rules for all canonical security_type values.
  - Enrichment job upserts profiles, handles errors per symbol.
  - Universe builder cache integration.
  - No live FMP calls.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
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
from alpha.jobs.contracts import JobContext
from alpha.jobs.runner import run_job
from alpha.jobs.security_type import (
    ADR,
    CLOSED_END_FUND,
    COMMON_STOCK,
    ETF,
    MUTUAL_FUND,
    BUSINESS_DEVELOPMENT_COMPANY,
    EXCHANGE_TRADED_DEBT,
    NON_COMMON_SERIES,
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
    profile_refresh_plan,
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

    def test_etf_from_name(self):
        st, reason = classify_security_type(_profile(
            symbol="WILD",
            company_name="VistaShares Animal Spirits Daily 2X Strategy ETF",
            is_etf=False,
        ))
        assert st == ETF
        assert reason == "name_contains:ETF"

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

    def test_preferred_from_five_char_p_symbol(self):
        # MPLXP-style: FMP serves the PARENT issuer's profile, so only the
        # symbol convention identifies the preferred.
        st, reason = classify_security_type(_profile(
            symbol="MPLXP",
            company_name="MPLX Lp",
            sector="Energy",
            industry="Oil & Gas Midstream",
            exchange="NYSE",
        ))
        assert st == PREFERRED
        assert reason == "symbol_suffix:fifth_char_P"

    def test_four_char_p_symbol_is_common(self):
        st, _ = classify_security_type(_profile(
            symbol="PUMP", company_name="ProPetro Holding Corp",
        ))
        assert st == COMMON_STOCK

    def test_five_char_non_p_symbol_is_common(self):
        st, _ = classify_security_type(_profile(
            symbol="GRPON", company_name="Acme Corp",
        ))
        assert st == COMMON_STOCK

    def test_five_char_p_symbol_with_stronger_evidence_keeps_earlier_type(self):
        # Name evidence outranks the last-resort symbol convention.
        st, _ = classify_security_type(_profile(
            symbol="ABCDP", company_name="Thunder Acquisition Corp",
        ))
        assert st == SPAC_OR_BLANK_CHECK

    def test_five_char_p_symbol_with_etf_sector_is_etf(self):
        # ETF sector/industry fallback outranks the symbol convention.
        st, reason = classify_security_type(_profile(
            symbol="ABCDP", company_name="Acme Strategy ETF Trust", sector="ETF",
        ))
        assert st == ETF
        assert reason == "sector_or_industry:ETF"

    def test_five_char_p_symbol_with_etf_industry_is_etf(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDP",
            company_name="Acme Strategy Trust",
            sector="Financial Services",
            industry="Exchange Traded Fund",
        ))
        assert st == ETF
        assert reason == "sector_or_industry:ETF"

    def test_five_char_p_symbol_with_is_fund_flag_is_mutual_fund(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDP", raw={"isFund": True},
        ))
        assert st == MUTUAL_FUND
        assert reason == "raw_flag:isFund"

    def test_five_char_p_symbol_with_is_adr_flag_is_adr(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDP", raw={"isAdr": True},
        ))
        assert st == ADR
        assert reason == "raw_flag:isAdr"

    def test_five_char_p_symbol_with_warrant_name_is_warrant(self):
        st, _ = classify_security_type(_profile(
            symbol="ABCDP", company_name="Acme Holdings Warrants",
        ))
        assert st == WARRANT

    def test_five_char_p_symbol_with_unit_name_is_unit(self):
        st, _ = classify_security_type(_profile(
            symbol="ABCDP",
            company_name="Acme Acquisition Units consisting of common stock",
        ))
        assert st == UNIT

    def test_five_char_p_symbol_with_right_name_is_right(self):
        st, _ = classify_security_type(_profile(
            symbol="ABCDP", company_name="Acme Subscription Rights",
        ))
        assert st == RIGHT

    def test_five_char_p_symbol_with_preferred_name_uses_name_reason(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDP", company_name="Acme Corp Preferred Series A",
        ))
        assert st == PREFERRED
        assert reason != "symbol_suffix:fifth_char_P"

    def test_five_char_p_symbol_actively_trading_still_preferred(self):
        # NHPAP/NHPBP: genuine preferreds with isActivelyTrading=True —
        # the P rule must stay ungated.
        st, reason = classify_security_type(_profile(
            symbol="NHPAP", company_name="National Holdings Corp",
            is_actively_trading=True,
        ))
        assert st == PREFERRED
        assert reason == "symbol_suffix:fifth_char_P"

    def test_inactive_o_suffix_is_preferred(self):
        # ZIONO-style: Series G preferred served with the parent profile.
        st, reason = classify_security_type(_profile(
            symbol="ZIONO",
            company_name="Zions Bancorporation, National Association",
            sector="Financial Services", industry="Banks - Regional",
            is_actively_trading=False,
        ))
        assert st == PREFERRED
        assert reason == "symbol_suffix:fifth_char_O+inactive"

    def test_inactive_l_suffix_is_non_common_series(self):
        # ZIONL-style: subordinated notes served with the parent profile;
        # L is the miscellaneous tape letter, so type stays non-specific.
        st, reason = classify_security_type(_profile(
            symbol="ZIONL",
            company_name="Zions Bancorporation N.A. - 6.9",
            sector="Financial Services", industry="Banks - Regional",
            is_actively_trading=False,
        ))
        assert st == NON_COMMON_SERIES
        assert reason == "symbol_suffix:fifth_char_L+inactive"

    def test_inactive_m_and_n_suffixes_are_preferred(self):
        for sym in ("NYMTM", "NYMTN"):
            st, _ = classify_security_type(_profile(
                symbol=sym, company_name="New York Mortgage Trust, Inc.",
                is_actively_trading=False,
            ))
            assert st == PREFERRED, sym

    def test_inactive_z_suffix_is_non_common_series(self):
        st, _ = classify_security_type(_profile(
            symbol="NYMTZ", company_name="New York Mortgage Trust, Inc.",
            is_actively_trading=False,
        ))
        assert st == NON_COMMON_SERIES

    def test_inactive_r_suffix_is_right(self):
        # CLRCR-style SPAC right whose profile name carries no evidence.
        st, _ = classify_security_type(_profile(
            symbol="CLRCR", company_name="ClimateRock",
            is_actively_trading=False,
        ))
        assert st == RIGHT

    def test_inactive_u_suffix_is_unit(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDU", company_name="Acme Parent Holdings",
            is_actively_trading=False,
        ))
        assert st == UNIT
        assert reason == "symbol_suffix:fifth_char_U+inactive"

    def test_inactive_w_suffix_is_warrant(self):
        st, reason = classify_security_type(_profile(
            symbol="ABCDW", company_name="Acme Parent Holdings",
            is_actively_trading=False,
        ))
        assert st == WARRANT
        assert reason == "symbol_suffix:fifth_char_W+inactive"

    def test_active_l_suffix_common_stays_common(self):
        # GOOGL: genuine common stock ending in L, actively trading —
        # the inactive gate must keep it common.
        st, reason = classify_security_type(_profile(
            symbol="GOOGL", company_name="Alphabet Inc.",
            sector="Technology", is_actively_trading=True,
        ))
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_active_o_suffix_common_stays_common(self):
        st, _ = classify_security_type(_profile(
            symbol="ABCDO", company_name="Acme Corp",
            is_actively_trading=True,
        ))
        assert st == COMMON_STOCK

    def test_inactive_suffix_from_raw_flag_only(self):
        # Gate must also fire when only the raw payload carries the flag.
        st, _ = classify_security_type(_profile(
            symbol="ZIONO", company_name="Zions Bancorporation",
            is_actively_trading=None, raw={"isActivelyTrading": False},
        ))
        assert st == PREFERRED

    def test_senior_notes_name_is_exchange_traded_debt(self):
        # FOSLL-style baby bond; name evidence works even when active.
        st, reason = classify_security_type(_profile(
            symbol="FOSLL",
            company_name="Fossil Group, Inc. 7% Senior Notes due 2026",
            is_actively_trading=True,
        ))
        assert st == EXCHANGE_TRADED_DEBT
        assert reason == "name_contains:NOTES"

    def test_coupon_notes_name_is_exchange_traded_debt(self):
        # CGBDL-style: "8.20% Notes ..." without the word SENIOR or DUE.
        st, _ = classify_security_type(_profile(
            symbol="CGBDL",
            company_name="Carlyle Secured Lending, Inc. 8.20% Notes",
        ))
        assert st == EXCHANGE_TRADED_DEBT

    def test_coupon_nt_name_is_exchange_traded_debt(self):
        # ECCX live name: "Eagle Point Credit Company Inc. 6.6875% NT 28".
        st, reason = classify_security_type(_profile(
            symbol="ECCX",
            company_name="Eagle Point Credit Company Inc. 6.6875% NT 28",
        ))
        assert st == EXCHANGE_TRADED_DEBT
        assert reason == "name_contains:NT"

    def test_notes_word_without_coupon_is_common(self):
        st, _ = classify_security_type(_profile(
            symbol="LNSC", company_name="Lotus Notes Software Corp",
        ))
        assert st == COMMON_STOCK

    def test_series_right_name_is_right(self):
        # AMPGR "Series A Right": actively trading, so only the name rule
        # can catch it.
        st, reason = classify_security_type(_profile(
            symbol="AMPGR",
            company_name="Amplitech Group, Inc. Series A Right",
            is_actively_trading=True,
        ))
        assert st == RIGHT
        assert reason == "name_contains:RIGHT"

    def test_closed_end_fund_from_asset_management_fund_template(self):
        # ASA-style: isFund=False but FMP fund-template description.
        st, reason = classify_security_type(_profile(
            symbol="ASA",
            company_name="ASA Gold and Precious Metals Limited",
            sector="Financial Services",
            industry="Asset Management",
            exchange="NYSE",
            raw={"description": (
                "ASA Gold and Precious Metals Limited is a publicly traded "
                "investment management firm. It operates as a global manager, "
                "primarily allocating capital in public equity markets. Its "
                "core investment strategy focuses on acquiring shares in "
                "companies engaged in the exploration of precious metals."
            )},
        ))
        assert st == CLOSED_END_FUND
        assert reason == "industry_description:ASSET_MANAGEMENT+FUND_TEMPLATE"

    def test_closed_end_fund_from_closed_end_equity_fund_description(self):
        # HQH-style live FMP profile: explicit closed-end equity fund.
        st, reason = classify_security_type(_profile(
            symbol="HQH",
            company_name="Abrdn Healthcare Investors",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "Abrdn Healthcare Investors is a closed-end equity fund, "
                "stewarded by abrdn Inc."
            )},
        ))
        assert st == CLOSED_END_FUND
        assert reason == "raw_description:CLOSED_END_EQUITY_FUND"

    def test_closed_end_fund_from_closed_ended_equity_mutual_fund_description(self):
        # HQL-style live FMP profile: explicit closed-ended equity mutual fund.
        st, reason = classify_security_type(_profile(
            symbol="HQL",
            company_name="Tekla Life Sciences Investors",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "Tekla Life Sciences Investors, a closed-ended equity "
                "mutual fund, is managed by Tekla Capital Management LLC."
            )},
        ))
        assert st == CLOSED_END_FUND
        assert reason == "raw_description:CLOSED_ENDED_EQUITY_MUTUAL_FUND"

    def test_known_cef_manual_override_for_blank_fmp_evidence(self):
        # CET/GAM live FMP profiles have Asset Management descriptions but no
        # closed-end phrase; SEC CIKs file NPORT-P/N-CEN/N-CSR.
        for symbol, name in (
            ("CET", "Central Securities Corp."),
            ("GAM", "General American Investors Company, Inc."),
        ):
            st, reason = classify_security_type(_profile(
                symbol=symbol,
                company_name=name,
                sector="Financial Services",
                industry="Asset Management",
                raw={"description": (
                    f"{name} functions as a publicly traded entity "
                    "specializing in investment management."
                )},
            ))
            assert st == CLOSED_END_FUND
            assert reason == "manual_override:known_cef"

    def test_eic_family_known_cef_override(self):
        # EIC is a registered closed-end fund; EICA/EICB/EICC are listed
        # term-preferred series served with the parent EIC profile.
        for symbol in ("EIC", "EICA", "EICB", "EICC"):
            st, reason = classify_security_type(_profile(
                symbol=symbol,
                company_name="Eagle Point Income Company Inc.",
                sector="Financial Services",
                industry="Asset Management",
                raw={"description": (
                    "Eagle Point Income Management manages funds through "
                    "separately managed accounts, and publicly traded "
                    "closed-end vehicles. The firm specializes in CLO debt."
                )},
            ))
            assert st == CLOSED_END_FUND
            assert reason == "manual_override:known_cef"

    @pytest.mark.parametrize(
        ("symbol", "description"),
        [
            (
                "DXYZ",
                "Destiny Tech100 Inc. is structured as a non-diversified, "
                "closed-end management company.",
            ),
            (
                "ECCC",
                "Eagle Point Credit Company Inc. is a closed-end investment "
                "fund established and overseen by Eagle Point Credit Management.",
            ),
            (
                "ECCF",
                "Eagle Point Credit Company Inc. is a closed-end investment "
                "fund, established in the United States on March 24, 2014.",
            ),
            (
                "ECCX",
                "Eagle Point Credit Company Inc. operates as a closed-end "
                "investment vehicle.",
            ),
            (
                "FSCO",
                "FS Credit Opportunities Corp. is an American closed-end "
                "fixed income fund.",
            ),
            (
                "SOR",
                "Source Capital, Inc. is a closed-end, balanced investment fund.",
            ),
        ],
    )
    def test_subject_position_closed_end_descriptions_are_cefs(
        self, symbol, description
    ):
        st, reason = classify_security_type(_profile(
            symbol=symbol,
            company_name=f"{symbol} Holdings",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": description},
        ))
        assert st == CLOSED_END_FUND
        assert reason == "raw_description:CLOSED_END_SUBJECT_POSITION"

    def test_closed_end_loans_lender_description_not_closed_end_fund(self):
        st, reason = classify_security_type(_profile(
            symbol="FISI",
            company_name="Financial Institutions, Inc.",
            sector="Financial Services",
            industry="Financial - Credit Services",
            raw={"description": (
                "Financial Institutions, Inc. offers home improvement loans, "
                "closed-end home equity loans, and home equity lines of credit."
            )},
        ))
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_closed_end_mutual_funds_managed_for_clients_not_closed_end_fund(self):
        st, reason = classify_security_type(_profile(
            symbol="BLK",
            company_name="BlackRock, Inc.",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "The firm manages open-end and closed-end mutual funds for "
                "institutional clients."
            )},
        ))
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_operating_asset_manager_not_closed_end_fund(self):
        # APAM-style operating manager: matches at most one template phrase.
        st, _ = classify_security_type(_profile(
            symbol="APAM",
            company_name="Artisan Partners Asset Management Inc.",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "Artisan Partners Asset Management Inc. is a publicly traded "
                "investment management firm that provides advisory services "
                "to mutual funds and institutional clients."
            )},
        ))
        assert st == COMMON_STOCK

    def test_operating_manager_with_closed_end_context_not_closed_end_fund(self):
        st, reason = classify_security_type(_profile(
            symbol="OMGR",
            company_name="Operating Manager Inc.",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "Operating Manager Inc. invests through separately managed "
                "accounts and publicly traded closed-end vehicles. The firm "
                "specializes in CLO strategy advisory services for clients."
            )},
        ))
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_operating_manager_with_both_template_phrases_not_closed_end_fund(self):
        # Adversarial: BOTH fund-template trigger phrases present, but the
        # client/advisory operating language blocks CEF classification.
        st, _ = classify_security_type(_profile(
            symbol="ACMM",
            company_name="Acme Capital Management Inc.",
            sector="Financial Services",
            industry="Asset Management",
            raw={"description": (
                "Acme Capital Management is an investment adviser allocating "
                "capital on behalf of institutional clients. The firm tailors "
                "each investment strategy to client mandates and provides "
                "advisory services and wealth management to high-net-worth "
                "individuals."
            )},
        ))
        assert st == COMMON_STOCK

    def test_fund_template_phrases_outside_asset_management_is_common(self):
        st, _ = classify_security_type(_profile(
            symbol="HOLDX",
            company_name="Acme Holdings Corp",
            sector="Industrials",
            industry="Conglomerates",
            raw={"description": (
                "Acme Holdings is a diversified operator, allocating capital "
                "across its subsidiaries; its investment strategy focuses on "
                "long-duration industrial assets."
            )},
        ))
        assert st == COMMON_STOCK

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
                raw={
                    "description": (
                        "Perceptive Capital Solutions Corp does not have "
                        "significant operations and intends to effect a business "
                        "combination with one or more businesses."
                    ),
                },
            ),
        )
        assert st == SPAC_OR_BLANK_CHECK
        assert reason == "industry_description:SHELL_COMPANIES+DOES_NOT_HAVE_SIGNIFICANT_OPERATIONS"

    def test_shell_company_industry_without_shell_description_is_not_spac(self):
        st, reason = classify_security_type(
            _profile(
                company_name="Central Plains Bancshares, Inc. Common Stock",
                sector="Financial Services",
                industry="Shell Companies",
                raw={
                    "description": (
                        "Central Plains Bancshares, Inc. focuses on providing "
                        "various banking products and services to retail customers, "
                        "and small and medium-sized commercial customers."
                    ),
                },
            )
        )
        assert st == COMMON_STOCK
        assert reason == "profile_fields_present"

    @pytest.mark.parametrize(
        ("name", "description"),
        [
            (
                "Aimei Health Technology Co., Ltd",
                "Aimei Health Technology Co., Ltd does not have significant operations. "
                "It intends to effect a merger, share exchange, asset acquisition, "
                "share purchase, recapitalization, reorganization, or similar business "
                "combination with one or more businesses or entities.",
            ),
            (
                "XFLH Capital Corporation",
                "XFLH Capital Corporation focuses on a merger, share exchange, "
                "asset acquisition, share purchase, reorganization or similar business "
                "combination with one or more businesses.",
            ),
        ],
    )
    def test_shell_company_industry_with_shell_description_is_spac(self, name, description):
        st, reason = classify_security_type(
            _profile(
                company_name=name,
                sector="Financial Services",
                industry="Shell Companies",
                raw={"description": description},
            )
        )
        assert st == SPAC_OR_BLANK_CHECK
        assert reason.startswith("industry_description:SHELL_COMPANIES+")

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
        assert EXCHANGE_TRADED_DEBT in NON_COMMON_TYPES
        assert NON_COMMON_SERIES in NON_COMMON_TYPES
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

    def test_profile_writes_flush_once_after_batch(self, db_session, monkeypatch):
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_company_profile.side_effect = lambda symbol: (
            _mock_profile_response(_profile(symbol))
        )
        flush_count = 0
        original_flush = db_session.flush

        def tracking_flush(*args, **kwargs):
            nonlocal flush_count
            flush_count += 1
            return original_flush(*args, **kwargs)

        monkeypatch.setattr(db_session, "flush", tracking_flush)
        job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=adapter,
            symbols=["ACME", "BETA", "GAMA"],
        )
        result = job.run(JobContext(
            job_id="job",
            job_run_id="run",
            started_at=_ts(),
        ))

        assert result.ok
        assert result.metrics["enriched_count"] == 3
        assert flush_count == 1

    def test_profile_refresh_plan_selects_only_stale_or_unresolved(self, db_session):
        asof = _ts()
        db_session.add_all([
            SecurityProfile(
                symbol="FRESH",
                security_type=COMMON_STOCK,
                refresh_status=REFRESH_STATUS_ENRICHED,
                classifier_version=CLASSIFIER_VERSION,
                last_refreshed_at=asof,
            ),
            SecurityProfile(
                symbol="STALE",
                security_type=COMMON_STOCK,
                refresh_status=REFRESH_STATUS_ENRICHED,
                classifier_version=CLASSIFIER_VERSION,
                last_refreshed_at=asof - timedelta(days=8),
            ),
            SecurityProfile(
                symbol="OLDV",
                security_type=COMMON_STOCK,
                refresh_status=REFRESH_STATUS_ENRICHED,
                classifier_version="old-version",
                last_refreshed_at=asof,
            ),
            SecurityProfile(
                symbol="NODATA",
                security_type=UNKNOWN,
                refresh_status=REFRESH_STATUS_NO_DATA,
                classifier_version=CLASSIFIER_VERSION,
                last_refreshed_at=asof,
            ),
        ])
        db_session.flush()

        refresh_symbols, metrics = profile_refresh_plan(
            db_session,
            ["fresh", "stale", "oldv", "nodata", "missing"],
            asof=asof,
            max_age_days=7,
        )

        assert refresh_symbols == ["MISSING", "NODATA", "OLDV", "STALE"]
        assert metrics["required_symbol_count"] == 5
        assert metrics["refresh_symbol_count"] == 4
        assert metrics["fresh_cached_count"] == 1
        assert metrics["missing_count"] == 1
        assert metrics["stale_count"] == 1
        assert metrics["classifier_version_mismatch_count"] == 1
        assert metrics["unresolved_count"] == 1

    def test_parallel_fetch_is_concurrent_and_metric_equivalent(self, db_session):
        symbols = ["ACME", "BETA", "GAMA", "DELT"]

        class SharedSlowState:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

        class SlowAdapter:
            def __init__(self, state=None):
                self.state = state or SharedSlowState()

            def get_company_profile(self, symbol):
                with self.state.lock:
                    self.state.active += 1
                    self.state.max_active = max(
                        self.state.max_active, self.state.active
                    )
                try:
                    time.sleep(0.05)
                    return _mock_profile_response(_profile(symbol))
                finally:
                    with self.state.lock:
                        self.state.active -= 1

        serial_adapter = SlowAdapter()
        serial_job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=serial_adapter,
            symbols=symbols,
            max_workers=1,
        )
        serial_start = time.perf_counter()
        serial_result = run_job(db_session, serial_job)
        serial_elapsed = time.perf_counter() - serial_start

        db_session.query(SecurityProfile).delete()
        db_session.flush()

        parallel_state = SharedSlowState()
        parallel_job = SecurityTypeEnrichmentJob(
            session=db_session,
            adapter=SlowAdapter(),
            symbols=symbols,
            max_workers=4,
            adapter_factory=lambda: SlowAdapter(parallel_state),
        )
        parallel_start = time.perf_counter()
        parallel_result = run_job(db_session, parallel_job)
        parallel_elapsed = time.perf_counter() - parallel_start

        assert serial_adapter.state.max_active == 1
        assert parallel_state.max_active > 1
        assert parallel_elapsed < serial_elapsed * 0.75
        assert parallel_result.metrics["enriched_count"] == serial_result.metrics["enriched_count"]
        assert parallel_result.metrics["failed_count"] == serial_result.metrics["failed_count"]
        assert parallel_result.metrics["security_type_counts"] == serial_result.metrics["security_type_counts"]

    def test_parallel_requires_adapter_factory(self, db_session):
        adapter = MagicMock(spec=FmpAdapter)

        with pytest.raises(ValueError, match="adapter_factory"):
            SecurityTypeEnrichmentJob(
                session=db_session,
                adapter=adapter,
                symbols=["ACME", "BETA"],
                max_workers=2,
            )

    def test_parallel_fetch_keeps_db_writes_on_main_thread(self, db_session):
        adapter_threads = set()

        class ThreadTrackingAdapter:
            def get_company_profile(self, symbol):
                adapter_threads.add(threading.get_ident())
                return _mock_profile_response(_profile(symbol))

        writer_threads = set()
        original_upsert = SecurityTypeEnrichmentJob._upsert_profile

        def tracking_upsert(self, *args, **kwargs):
            writer_threads.add(threading.get_ident())
            return original_upsert(self, *args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SecurityTypeEnrichmentJob, "_upsert_profile", tracking_upsert)
            job = SecurityTypeEnrichmentJob(
                session=db_session,
                adapter=ThreadTrackingAdapter(),
                symbols=["ACME", "BETA", "GAMA", "DELT"],
                max_workers=4,
                adapter_factory=ThreadTrackingAdapter,
            )
            result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["enriched_count"] == 4
        assert writer_threads == {threading.get_ident()}
        assert adapter_threads

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
                exclusion_reason="price_below_2",
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
        assert result.metrics["security_profile_coverage_required_count"] == 1
        assert result.metrics["security_profile_coverage_headroom_count"] == 0
        assert result.metrics["security_profile_coverage_shortfall_count"] == 0
        assert result.metrics["security_profile_unenriched_required_count"] == 0
        assert result.input_hashes["security_profile_cache"]
        scan = db_session.query(UniverseScan).one()
        assert scan.security_profile_cache_hash == result.input_hashes["security_profile_cache"]

    def test_non_common_type_excludes_with_reason(self, db_session):
        db_session.add(SecurityProfile(
            symbol="BADF", security_type=MUTUAL_FUND,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
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
        assert result.metrics["security_profile_coverage_required_count"] == 1
        assert result.metrics["security_profile_coverage_headroom_count"] == -1
        assert result.metrics["security_profile_coverage_shortfall_count"] == 1
        assert result.metrics["security_profile_unenriched_required_count"] == 1

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
        assert result.metrics["security_profile_coverage_required_count"] == 1
        assert result.metrics["security_profile_coverage_headroom_count"] == -1
        assert result.metrics["security_profile_coverage_shortfall_count"] == 1
        assert result.metrics["security_profile_unenriched_required_count"] == 1

        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "failed"
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_required_security_profile_coverage_allows_complete_cache(self, db_session):
        db_session.add(SecurityProfile(
            symbol="ACME", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
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
        assert result.metrics["security_profile_coverage_required_count"] == 1
        assert result.metrics["security_profile_coverage_headroom_count"] == 0
        assert result.metrics["security_profile_coverage_shortfall_count"] == 0
        assert result.metrics["security_profile_unenriched_required_count"] == 0
        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "finished"
        assert db_session.query(CanonicalUniverseScan).count() == 1

    def test_unknown_type_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="MYSTK", security_type=UNKNOWN,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_NO_DATA,
            classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
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

    def test_old_classifier_version_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="OLDV", security_type=COMMON_STOCK,
            last_refreshed_at=_ts(), refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version="security_type_v0",
        ))
        db_session.flush()

        resp = AdapterResponse(data=[_stock("OLDV")], lineage=_mock_lineage_screener())
        job = UniverseBuilderJob(
            session=db_session,
            screener_response=resp,
            require_security_profile_cache=True,
            min_security_profile_coverage=1.0,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "failed"
        assert result.metrics["failure_stage"] == "security_profile_coverage"
        assert result.metrics["security_profile_required_count"] == 1
        assert result.metrics["security_profile_enriched_count"] == 0
        assert (
            result.metrics["security_profile_classifier_version_mismatch_count"]
            == 1
        )
        assert result.metrics["security_profile_coverage_shortfall_count"] == 1
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "OLDV"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_profile_classifier_version_mismatch"
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_stale_profile_excludes_clean_symbol(self, db_session):
        db_session.add(SecurityProfile(
            symbol="OLD", security_type=COMMON_STOCK,
            last_refreshed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            refresh_status=REFRESH_STATUS_ENRICHED,
            classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
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
                classifier_version=CLASSIFIER_VERSION,
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
            classifier_version=CLASSIFIER_VERSION,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("PENY", price=1.0)],  # fails price_below_2
            lineage=_mock_lineage_screener(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "PENY"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "price_below_2"


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
