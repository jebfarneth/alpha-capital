"""Mark-don't-delete ML exclusion metadata for historical ML corpora.

The 2024-2026 historical M4 corpus and the I12 historical corpus source their
tickers through historical universe reconstruction, which (unlike the live
universe builder) carries no security-type exclusion. A read-only
quantification with the security_type_v7 classifier records, for EVERY ticker
in the M4 corpus artifact or included by HUR, whether it is a common stock or a
non-common instrument (SPAC shell, ETF, mutual fund, ADR, preferred,
exchange-traded debt, warrant/unit/right series, BDC, CEF).

Policy: signal_registry rows are NEVER deleted or mutated for this. The
classification artifact in alpha/ml/data/ is the durable exclusion record;
training-set assembly (the ML dataset manifest) must consult
``is_ml_excluded`` / ``non_common_tickers`` when selecting rows.

Fail-closed guarantees, all enforced at load time:
- CSV sha256 must match the sidecar metadata.
- Metadata classifier version must match the live classifier, so a
  classifier bump forces regeneration instead of serving a stale list.
- Every row must carry non-blank ticker/security_type/reason fields, a
  recognized security type (common or a member of NON_COMMON_TYPES), and a
  positive signal count; unresolved markers such as ``no_profile``/``unknown``
  are rejected.
- The CSV must be strictly ticker-sorted and duplicate-free.
- Recomputed totals (corpus/excluded signals and tickers, excluded-by-type
  counts, excluded-by-signal type/reason counts, and excluded percentage) must
  match the metadata.
- ``is_ml_excluded`` RAISES for tickers absent from the artifact: a miss
  means the caller is outside the artifact's corpus window (drift, typo,
  or new selection), which must force regeneration, not pass as clean.

PIT caveat (also recorded in the artifact metadata): the security type is
the classified_asof-day profile applied retroactively. Names that have since
converted (e.g. de-SPACed into operating companies) classify common_stock
today, so early-window SPAC contamination is understated. Types are NOT
point-in-time as of each signal's trading_date.

Provenance: the artifact is produced by the committed generator
``alpha/ml/generate_m4_security_type_artifact.py`` (read-only DB + FMP).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, NamedTuple

from alpha.jobs.security_type import (
    CLASSIFIER_VERSION,
    COMMON_STOCK,
    NON_COMMON_TYPES,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSIFICATION_ARTIFACT_PATH = _DATA_DIR / "m4_corpus_security_types_v8.csv"
CLASSIFICATION_METADATA_PATH = _DATA_DIR / "m4_corpus_security_types_v8.meta.json"

VALID_ARTIFACT_TYPES = frozenset({COMMON_STOCK}) | NON_COMMON_TYPES


class SecurityTypeClassification(NamedTuple):
    ticker: str
    security_type: str
    reason: str
    signals: int

    @property
    def ml_excluded(self) -> bool:
        return self.security_type in NON_COMMON_TYPES


class ExclusionArtifactError(RuntimeError):
    """The exclusion artifact is missing, drifted, stale, or malformed."""


@lru_cache(maxsize=1)
def load_artifact_metadata() -> dict:
    if not CLASSIFICATION_METADATA_PATH.exists():
        raise ExclusionArtifactError(
            f"missing exclusion artifact metadata: {CLASSIFICATION_METADATA_PATH}"
        )
    with open(CLASSIFICATION_METADATA_PATH, "r") as f:
        return json.load(f)


def _verify_artifact() -> dict:
    meta = load_artifact_metadata()
    if not CLASSIFICATION_ARTIFACT_PATH.exists():
        raise ExclusionArtifactError(
            f"missing exclusion artifact: {CLASSIFICATION_ARTIFACT_PATH}"
        )
    digest = hashlib.sha256(CLASSIFICATION_ARTIFACT_PATH.read_bytes()).hexdigest()
    if digest != meta.get("artifact_sha256"):
        raise ExclusionArtifactError(
            "exclusion artifact sha256 mismatch: artifact drifted from metadata; "
            "regenerate both with alpha/ml/generate_m4_security_type_artifact.py"
        )
    if meta.get("classifier_version") != CLASSIFIER_VERSION:
        raise ExclusionArtifactError(
            f"exclusion artifact built with {meta.get('classifier_version')!r} but live "
            f"classifier is {CLASSIFIER_VERSION!r}; regenerate the artifact"
        )
    return meta


def _validate_against_metadata(
    meta: dict, records: Dict[str, SecurityTypeClassification]
) -> None:
    totals = meta.get("totals", {})
    excluded = [r for r in records.values() if r.ml_excluded]
    checks = {
        "corpus_tickers": len(records),
        "corpus_signals": sum(r.signals for r in records.values()),
        "excluded_tickers": len(excluded),
        "excluded_signals": sum(r.signals for r in excluded),
    }
    for key, computed in checks.items():
        if totals.get(key) != computed:
            raise ExclusionArtifactError(
                f"exclusion artifact metadata mismatch: totals[{key!r}] = "
                f"{totals.get(key)!r} but artifact computes {computed!r}"
            )
    excluded_signal_pct = round(
        100 * checks["excluded_signals"] / checks["corpus_signals"], 2
    )
    if totals.get("excluded_signal_pct") != excluded_signal_pct:
        raise ExclusionArtifactError(
            "exclusion artifact metadata mismatch: totals['excluded_signal_pct'] "
            f"= {totals.get('excluded_signal_pct')!r} but artifact computes "
            f"{excluded_signal_pct!r}"
        )
    ticker_by_type = Counter(r.security_type for r in excluded)
    meta_ticker_by_type = meta.get("excluded_tickers_by_type")
    if meta_ticker_by_type is not None and dict(ticker_by_type) != meta_ticker_by_type:
        raise ExclusionArtifactError(
            "exclusion artifact metadata mismatch: excluded_tickers_by_type "
            f"= {meta_ticker_by_type!r} but artifact computes {dict(ticker_by_type)!r}"
        )
    signals_by_type = Counter()
    signals_by_reason = Counter()
    for rec in excluded:
        signals_by_type[rec.security_type] += rec.signals
        signals_by_reason[rec.reason] += rec.signals
    meta_signals_by_type = meta.get("excluded_signals_by_type")
    if meta_signals_by_type is not None and dict(signals_by_type) != meta_signals_by_type:
        raise ExclusionArtifactError(
            "exclusion artifact metadata mismatch: excluded_signals_by_type "
            f"= {meta_signals_by_type!r} but artifact computes "
            f"{dict(signals_by_type)!r}"
        )
    meta_signals_by_reason = meta.get("excluded_signals_by_reason")
    if (
        meta_signals_by_reason is not None
        and dict(signals_by_reason) != meta_signals_by_reason
    ):
        raise ExclusionArtifactError(
            "exclusion artifact metadata mismatch: excluded_signals_by_reason "
            f"= {meta_signals_by_reason!r} but artifact computes "
            f"{dict(signals_by_reason)!r}"
        )

    # The CSV is ticker-level and carries no month dimension. We can still
    # validate month metadata is internally consistent with top-level totals.
    month_rows = meta.get("excluded_signals_by_month")
    if month_rows is not None:
        month_excluded = sum(row.get("excluded", 0) for row in month_rows.values())
        month_total = sum(row.get("total", 0) for row in month_rows.values())
        if month_excluded != checks["excluded_signals"]:
            raise ExclusionArtifactError(
                "exclusion artifact metadata mismatch: excluded_signals_by_month "
                f"excluded sum = {month_excluded!r} but totals excluded_signals "
                f"= {checks['excluded_signals']!r}"
            )
        if month_total != checks["corpus_signals"]:
            raise ExclusionArtifactError(
                "exclusion artifact metadata mismatch: excluded_signals_by_month "
                f"total sum = {month_total!r} but totals corpus_signals "
                f"= {checks['corpus_signals']!r}"
            )


@lru_cache(maxsize=1)
def load_classifications() -> Dict[str, SecurityTypeClassification]:
    """Ticker -> classification for every ticker covered by the corpus artifact."""
    meta = _verify_artifact()
    out: Dict[str, SecurityTypeClassification] = {}
    previous_ticker: str | None = None
    with open(CLASSIFICATION_ARTIFACT_PATH, "r", newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip()
            security_type = (row.get("security_type") or "").strip()
            reason = (row.get("reason") or "").strip()
            if not ticker:
                raise ExclusionArtifactError("blank ticker in exclusion artifact")
            if not security_type:
                raise ExclusionArtifactError(
                    f"blank security_type in exclusion artifact for {ticker!r}"
                )
            if not reason:
                raise ExclusionArtifactError(
                    f"blank reason in exclusion artifact for {ticker!r}"
                )
            try:
                signals = int(row["signals"])
            except (TypeError, ValueError):
                raise ExclusionArtifactError(
                    f"malformed signals count for {ticker!r}: "
                    f"{row.get('signals')!r}"
                )
            rec = SecurityTypeClassification(
                ticker=ticker,
                security_type=security_type,
                reason=reason,
                signals=signals,
            )
            if rec.security_type not in VALID_ARTIFACT_TYPES:
                raise ExclusionArtifactError(
                    f"unrecognized/unresolved security_type {rec.security_type!r} "
                    f"for {rec.ticker!r}; the artifact must contain only common "
                    "stock or NON_COMMON_TYPES members"
                )
            if rec.signals <= 0:
                raise ExclusionArtifactError(
                    f"nonpositive signal count for {rec.ticker!r}: {rec.signals}"
                )
            if rec.ticker in out:
                raise ExclusionArtifactError(
                    f"duplicate ticker in exclusion artifact: {rec.ticker}"
                )
            if previous_ticker is not None and rec.ticker <= previous_ticker:
                raise ExclusionArtifactError(
                    "exclusion artifact tickers must be strictly sorted: "
                    f"{rec.ticker!r} follows {previous_ticker!r}"
                )
            out[rec.ticker] = rec
            previous_ticker = rec.ticker
    if not out:
        raise ExclusionArtifactError("exclusion artifact is empty")
    _validate_against_metadata(meta, out)
    return out


@lru_cache(maxsize=1)
def non_common_tickers() -> FrozenSet[str]:
    """Tickers whose M4 corpus signals must be excluded from training sets."""
    return frozenset(
        t for t, rec in load_classifications().items() if rec.ml_excluded
    )


def is_ml_excluded(ticker: str) -> bool:
    """True if the ticker's corpus signals are non-common (mark-don't-delete).

    Fail-closed: raises ExclusionArtifactError for tickers absent from the
    artifact. The artifact covers every ticker with >=1 M4 signal in its
    corpus_window, so a miss means the caller is selecting outside the
    artifact's scope (corpus drift, wrong window, typo) and the artifact
    must be regenerated — unknown names must never silently pass as clean.
    """
    records = load_classifications()
    if ticker not in records:
        raise ExclusionArtifactError(
            f"ticker {ticker!r} is not covered by the exclusion artifact "
            f"(corpus window {load_artifact_metadata().get('corpus_window')}); "
            "regenerate the artifact before selecting it for training"
        )
    return records[ticker].ml_excluded
