"""Tests for the mark-don't-delete ML exclusion artifact + loader."""
from __future__ import annotations

import hashlib
import json

import pytest

from alpha.jobs.security_type import CLASSIFIER_VERSION, NON_COMMON_TYPES
from alpha.ml import security_type_exclusions as ste


@pytest.fixture(autouse=True)
def _clear_caches():
    ste.load_artifact_metadata.cache_clear()
    ste.load_classifications.cache_clear()
    ste.non_common_tickers.cache_clear()
    yield
    ste.load_artifact_metadata.cache_clear()
    ste.load_classifications.cache_clear()
    ste.non_common_tickers.cache_clear()


def _write_pair(tmp_path, monkeypatch, csv_text, meta):
    """Install a synthetic artifact CSV + metadata with a consistent hash."""
    art = tmp_path / "artifact.csv"
    art.write_text(csv_text)
    meta = dict(meta)
    meta.setdefault("classifier_version", CLASSIFIER_VERSION)
    meta["artifact_sha256"] = hashlib.sha256(art.read_bytes()).hexdigest()
    meta_path = tmp_path / "artifact.meta.json"
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(ste, "CLASSIFICATION_ARTIFACT_PATH", art)
    monkeypatch.setattr(ste, "CLASSIFICATION_METADATA_PATH", meta_path)


_GOOD_CSV = (
    "ticker,security_type,reason,signals\n"
    "AAAA,common_stock,profile_fields_present,10\n"
    "BBBB,etf,is_etf=True,5\n"
)
_GOOD_TOTALS = {
    "totals": {
        "corpus_tickers": 2,
        "corpus_signals": 15,
        "excluded_tickers": 1,
        "excluded_signals": 5,
        "excluded_signal_pct": 33.33,
    },
    "excluded_tickers_by_type": {"etf": 1},
    "excluded_signals_by_type": {"etf": 5},
    "excluded_signals_by_reason": {"is_etf=True": 5},
    "excluded_signals_by_month": {"2024-01": {"excluded": 5, "total": 15}},
}


class TestArtifactIntegrity:
    def test_artifact_files_exist(self):
        assert ste.CLASSIFICATION_ARTIFACT_PATH.exists()
        assert ste.CLASSIFICATION_METADATA_PATH.exists()

    def test_sha256_matches_metadata(self):
        meta = ste.load_artifact_metadata()
        digest = hashlib.sha256(
            ste.CLASSIFICATION_ARTIFACT_PATH.read_bytes()
        ).hexdigest()
        assert digest == meta["artifact_sha256"]

    def test_metadata_classifier_version_matches_live(self):
        meta = ste.load_artifact_metadata()
        assert meta["classifier_version"] == CLASSIFIER_VERSION

    def test_metadata_documents_pit_caveat_window_and_generator(self):
        meta = ste.load_artifact_metadata()
        assert "retroactively" in meta["pit_caveat"]
        assert meta["corpus_window"]["pattern_id"] == "M4"
        assert meta["corpus_window"]["trading_date_min"] == "2024-01-01"
        assert "generate_m4_security_type_artifact" in meta["generator"]
        assert "signal_registry" in meta["corpus_query"]

    def test_metadata_totals_consistent_with_artifact(self):
        meta = ste.load_artifact_metadata()
        recs = ste.load_classifications()
        assert meta["totals"]["corpus_tickers"] == len(recs)
        assert meta["totals"]["corpus_signals"] == sum(
            r.signals for r in recs.values()
        )
        assert meta["totals"]["excluded_tickers"] == len(ste.non_common_tickers())
        assert meta["totals"]["excluded_signals"] == sum(
            r.signals for r in recs.values() if r.ml_excluded
        )


class TestLoader:
    def test_every_ticker_classified_no_unresolved(self):
        recs = ste.load_classifications()
        types = {r.security_type for r in recs.values()}
        assert "no_profile" not in types
        assert "unknown" not in types
        assert types <= ste.VALID_ARTIFACT_TYPES

    def test_exclusion_membership_derived_from_non_common_types(self):
        recs = ste.load_classifications()
        expected = {
            t for t, r in recs.items() if r.security_type in NON_COMMON_TYPES
        }
        assert ste.non_common_tickers() == expected

    def test_known_exclusions_present(self):
        # Catches verified live 2026-06-10 (v4 audit + v5 audit findings).
        recs = ste.load_classifications()
        assert recs["MPLXP"].security_type == "preferred"
        assert recs["MPLXP"].reason == "symbol_suffix:fifth_char_P"
        assert recs["ASA"].security_type == "closed_end_fund"
        assert ste.is_ml_excluded("MPLXP")
        assert ste.is_ml_excluded("ASA")

    def test_zion_series_listings_excluded(self):
        # Audit finding: ZIONL (subordinated notes) and ZIONO (Series G
        # preferred) must not pass as common stock.
        recs = ste.load_classifications()
        assert ste.is_ml_excluded("ZIONL")
        assert ste.is_ml_excluded("ZIONO")
        assert recs["ZIONO"].security_type == "preferred"
        assert recs["ZIONL"].security_type in {
            "non_common_series",
            "exchange_traded_debt",
        }

    def test_known_cefs_excluded(self):
        recs = ste.load_classifications()
        for ticker in (
            "CET", "GAM", "HQH", "HQL", "TY",
            "DXYZ", "ECCC", "ECCF", "FSCO", "SOR",
            "EIC", "EICA", "EICB", "EICC",
        ):
            assert recs[ticker].security_type == "closed_end_fund"
            assert ste.is_ml_excluded(ticker)

    def test_eccx_nt_notes_excluded(self):
        recs = ste.load_classifications()
        assert recs["ECCX"].security_type == "exchange_traded_debt"
        assert recs["ECCX"].reason == "name_contains:NT"
        assert ste.is_ml_excluded("ECCX")

    def test_wild_delisted_etf_fallback_excluded(self):
        recs = ste.load_classifications()
        assert recs["WILD"].security_type == "etf"
        assert recs["WILD"].reason == "delisted_name:name_contains:ETF"
        assert ste.is_ml_excluded("WILD")

    def test_baby_bond_and_series_suffix_cohort_excluded(self):
        # The broader adversarial set from the audit, all verified series
        # instruments of 4-letter parents.
        for ticker in (
            "ATCOL", "CGBDL", "GAINL", "NYMTL", "NYMTM", "NYMTN", "NYMTZ",
            "FOSLL", "HROWL", "HROWM", "METCL", "NEWTL", "LANDM", "WTFCM",
            "CCLDO", "ESGRO", "FTAIO", "MBINO", "RILYO", "GLADZ", "RWAYZ",
            "HYMCL", "AMPGZ", "AMPGR", "CHKEZ", "CLRCR",
        ):
            assert ste.is_ml_excluded(ticker), ticker

    def test_specific_common_stocks_not_excluded(self):
        for ticker in ("AADI", "ZVRA"):
            assert not ste.is_ml_excluded(ticker), ticker

    def test_common_stock_not_excluded(self):
        recs = ste.load_classifications()
        common = next(
            t for t, r in recs.items() if r.security_type == "common_stock"
        )
        assert not ste.is_ml_excluded(common)


class TestFailClosed:
    def test_unknown_ticker_raises(self):
        # Fail-closed: out-of-scope tickers must never pass as clean.
        with pytest.raises(ste.ExclusionArtifactError, match="not covered"):
            ste.is_ml_excluded("ZZZZZZZZ")

    def test_artifact_drift_raises(self, tmp_path, monkeypatch):
        drifted = tmp_path / "drifted.csv"
        drifted.write_bytes(
            ste.CLASSIFICATION_ARTIFACT_PATH.read_bytes() + b"X,common_stock,x,1\n"
        )
        monkeypatch.setattr(ste, "CLASSIFICATION_ARTIFACT_PATH", drifted)
        with pytest.raises(ste.ExclusionArtifactError, match="sha256 mismatch"):
            ste.load_classifications()

    def test_stale_classifier_version_raises(self, tmp_path, monkeypatch):
        meta = json.loads(ste.CLASSIFICATION_METADATA_PATH.read_text())
        meta["classifier_version"] = "security_type_v3"
        stale = tmp_path / "stale.meta.json"
        stale.write_text(json.dumps(meta))
        monkeypatch.setattr(ste, "CLASSIFICATION_METADATA_PATH", stale)
        with pytest.raises(ste.ExclusionArtifactError, match="regenerate"):
            ste.load_classifications()

    def test_missing_artifact_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ste, "CLASSIFICATION_ARTIFACT_PATH", tmp_path / "absent.csv"
        )
        with pytest.raises(ste.ExclusionArtifactError, match="missing"):
            ste.load_classifications()

    def test_missing_metadata_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ste, "CLASSIFICATION_METADATA_PATH", tmp_path / "absent.meta.json"
        )
        with pytest.raises(ste.ExclusionArtifactError, match="missing"):
            ste.load_classifications()

    def test_unresolved_type_in_artifact_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,no_profile,fetch_failed,10\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="unrecognized"):
            ste.load_classifications()

    def test_unknown_type_in_artifact_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,unknown,insufficient_profile_data,10\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="unrecognized"):
            ste.load_classifications()

    def test_blank_ticker_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            " ,common_stock,profile_fields_present,10\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="blank ticker"):
            ste.load_classifications()

    def test_blank_reason_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,common_stock, ,10\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="blank reason"):
            ste.load_classifications()

    def test_unsorted_csv_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "BBBB,etf,is_etf=True,5\n"
            "AAAA,common_stock,profile_fields_present,10\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, _GOOD_TOTALS)
        with pytest.raises(ste.ExclusionArtifactError, match="strictly sorted"):
            ste.load_classifications()

    def test_duplicate_ticker_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,common_stock,profile_fields_present,10\n"
            "AAAA,etf,is_etf=True,5\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, _GOOD_TOTALS)
        with pytest.raises(ste.ExclusionArtifactError, match="duplicate ticker"):
            ste.load_classifications()

    def test_nonpositive_signals_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,common_stock,profile_fields_present,0\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="nonpositive"):
            ste.load_classifications()

    def test_malformed_signals_raises(self, tmp_path, monkeypatch):
        csv_text = (
            "ticker,security_type,reason,signals\n"
            "AAAA,common_stock,profile_fields_present,abc\n"
        )
        _write_pair(tmp_path, monkeypatch, csv_text, {})
        with pytest.raises(ste.ExclusionArtifactError, match="malformed"):
            ste.load_classifications()

    def test_metadata_totals_mismatch_raises(self, tmp_path, monkeypatch):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["totals"]["excluded_signals"] = 999
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(ste.ExclusionArtifactError, match="metadata mismatch"):
            ste.load_classifications()

    def test_metadata_by_type_mismatch_raises(self, tmp_path, monkeypatch):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["excluded_tickers_by_type"] = {"etf": 2}
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(ste.ExclusionArtifactError, match="by_type"):
            ste.load_classifications()

    def test_metadata_excluded_signals_by_type_mismatch_raises(
        self, tmp_path, monkeypatch
    ):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["excluded_signals_by_type"] = {"etf": 999}
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(ste.ExclusionArtifactError, match="excluded_signals_by_type"):
            ste.load_classifications()

    def test_metadata_excluded_signals_by_reason_mismatch_raises(
        self, tmp_path, monkeypatch
    ):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["excluded_signals_by_reason"] = {"is_etf=True": 999}
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(
            ste.ExclusionArtifactError, match="excluded_signals_by_reason"
        ):
            ste.load_classifications()

    def test_metadata_excluded_signals_by_month_inconsistency_raises(
        self, tmp_path, monkeypatch
    ):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["excluded_signals_by_month"] = {"2024-01": {"excluded": 4, "total": 15}}
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(
            ste.ExclusionArtifactError, match="excluded_signals_by_month"
        ):
            ste.load_classifications()

    def test_metadata_excluded_signal_pct_mismatch_raises(self, tmp_path, monkeypatch):
        bad = json.loads(json.dumps(_GOOD_TOTALS))
        bad["totals"]["excluded_signal_pct"] = 99.99
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, bad)
        with pytest.raises(ste.ExclusionArtifactError, match="excluded_signal_pct"):
            ste.load_classifications()

    def test_consistent_synthetic_pair_loads(self, tmp_path, monkeypatch):
        _write_pair(tmp_path, monkeypatch, _GOOD_CSV, _GOOD_TOTALS)
        recs = ste.load_classifications()
        assert recs["BBBB"].ml_excluded and not recs["AAAA"].ml_excluded
