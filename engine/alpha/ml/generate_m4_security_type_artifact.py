"""Generate the historical corpus security-type classification artifact.

Produces alpha/ml/data/m4_corpus_security_types_v8.csv and its sidecar
metadata for consumption by alpha.ml.security_type_exclusions. The database is
only ever read (the previous v7 artifact, historical universe reconstruction,
and the fmp_delisted_companies directory); classification inputs come from the
live FMP profile endpoint. The only writes are the two v8 artifact files.

Resolution order per ticker:
1. FMP /stable/profile, classified with the live classifier
   (alpha.jobs.security_type.classify_security_type), with per-ticker
   retries so transient fetch failures cannot masquerade as delistings.
2. For tickers with no live profile: the fmp_delisted_companies directory's
   company_name/exchange, classified through the same name rules.
3. Any ticker still unresolved fails the run — the artifact contract is
   zero unknown/no_profile rows (the loader rejects them).

Usage (from engine/):
    uv run python -m alpha.ml.generate_m4_security_type_artifact
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

from sqlalchemy import create_engine, text

from alpha.data.fmp import FmpCompanyProfile
from alpha.jobs.security_type import (
    CLASSIFIER_VERSION,
    NON_COMMON_TYPES,
    classify_security_type,
)
from alpha.ml.security_type_exclusions import (
    CLASSIFICATION_ARTIFACT_PATH,
    CLASSIFICATION_METADATA_PATH,
)

CORPUS_WINDOW = {
    "pattern_id": "M4+HUR",
    "trading_date_min": "2024-01-01",
    "trading_date_max": "2026-06-30",
}
PRIOR_ARTIFACT_PATH = (
    Path(__file__).resolve().parent / "data" / "m4_corpus_security_types_v7.csv"
)
PRIOR_METADATA_PATH = (
    Path(__file__).resolve().parent / "data" / "m4_corpus_security_types_v7.meta.json"
)
HUR_INCLUDED_QUERY = """select normalized_symbol, count(*)
    from historical_universe_reconstructions
    where inclusion_status = 'included'
      and normalized_symbol is not null
    group by 1"""
PROFILE_FETCH_ATTEMPTS = 3
PROFILE_FETCH_BACKOFF_SECONDS = 2.0


def _fetch_profile(ticker: str, api_key: str) -> Optional[dict]:
    url = (
        "https://financialmodelingprep.com/stable/profile"
        f"?symbol={ticker}&apikey={api_key}"
    )
    for attempt in range(PROFILE_FETCH_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            return data[0] if isinstance(data, list) and data else None
        except Exception:
            if attempt + 1 < PROFILE_FETCH_ATTEMPTS:
                time.sleep(PROFILE_FETCH_BACKOFF_SECONDS)
    return None


def _classify_live(ticker: str, raw: dict) -> Tuple[str, str]:
    profile = FmpCompanyProfile(
        symbol=ticker,
        company_name=raw.get("companyName") or "",
        market_cap=raw.get("marketCap"),
        sector=raw.get("sector"),
        industry=raw.get("industry"),
        exchange=raw.get("exchangeShortName") or raw.get("exchange"),
        country=raw.get("country"),
        is_etf=raw.get("isEtf"),
        is_actively_trading=raw.get("isActivelyTrading"),
        ipo_date=raw.get("ipoDate"),
        raw=raw,
    )
    return classify_security_type(profile, raw_json=raw)


def _classify_from_delisted_directory(
    conn, tickers: list
) -> Dict[str, Tuple[str, str]]:
    rows = conn.execute(
        text(
            """select normalized_symbol, company_name, exchange
            from fmp_delisted_companies where normalized_symbol = any(:t)"""
        ),
        {"t": tickers},
    ).fetchall()
    out: Dict[str, Tuple[str, str]] = {}
    for symbol, name, exchange in rows:
        profile = FmpCompanyProfile(
            symbol=symbol,
            company_name=name or "",
            market_cap=None,
            sector=None,
            industry=None,
            exchange=exchange,
            country=None,
            is_etf=None,
            is_actively_trading=None,
            ipo_date=None,
            raw=None,
        )
        security_type, reason = classify_security_type(profile, raw_json={})
        out[symbol] = (security_type, f"delisted_name:{reason}")
    return out


def _load_prior_artifact() -> Tuple[Counter, Dict[str, Tuple[str, str]]]:
    if not PRIOR_ARTIFACT_PATH.exists():
        return Counter(), {}
    counts: Counter = Counter()
    classified: Dict[str, Tuple[str, str]] = {}
    with open(PRIOR_ARTIFACT_PATH, "r", newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            counts[ticker] += int(row["signals"])
            classified[ticker] = (
                (row.get("security_type") or "").strip(),
                (row.get("reason") or "").strip(),
            )
    return counts, classified


def generate() -> dict:
    api_key = os.environ["FMP_API_KEY"]
    engine = create_engine(os.environ["DATABASE_URL"])
    counts, _prior_classified = _load_prior_artifact()
    prior_artifact_tickers = set(counts)
    with engine.connect() as conn:
        hur_rows = conn.execute(text(HUR_INCLUDED_QUERY)).fetchall()
        for ticker, n in hur_rows:
            normalized = str(ticker).strip().upper()
            if normalized:
                counts[normalized] += int(n)
        tickers = sorted(counts)
        tickers_to_classify = list(tickers)
        print(
            "artifact tickers: "
            f"{len(tickers)}, coverage rows: {sum(counts.values())}, "
            f"prior artifact tickers: {len(prior_artifact_tickers)}, "
            f"classifications needed: {len(tickers_to_classify)}"
        )

        with ThreadPoolExecutor(max_workers=10) as ex:
            profiles = dict(
                ex.map(lambda t: (t, _fetch_profile(t, api_key)), tickers_to_classify)
            )

        classified: Dict[str, Tuple[str, str]] = {}
        missing = []
        for ticker in tickers_to_classify:
            raw = profiles[ticker]
            if raw is None:
                missing.append(ticker)
            else:
                classified[ticker] = _classify_live(ticker, raw)
        print(f"live profiles: {len(classified)}, missing: {len(missing)}")

        if missing:
            classified.update(_classify_from_delisted_directory(conn, missing))

    unresolved = [t for t in tickers if t not in classified]
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} tickers unresolved after live + delisted "
            f"directory resolution: {unresolved}; artifact not written"
        )

    rows = [
        {
            "ticker": t,
            "security_type": classified[t][0],
            "reason": classified[t][1],
            "signals": counts[t],
        }
        for t in tickers
    ]
    CLASSIFICATION_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASSIFICATION_ARTIFACT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ticker", "security_type", "reason", "signals"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    sha = hashlib.sha256(CLASSIFICATION_ARTIFACT_PATH.read_bytes()).hexdigest()

    excluded = [r for r in rows if r["security_type"] in NON_COMMON_TYPES]
    total_signals = sum(counts.values())
    excluded_signals = sum(r["signals"] for r in excluded)
    by_type: Counter = Counter()
    by_reason: Counter = Counter()
    for r in excluded:
        by_type[r["security_type"]] += r["signals"]
        by_reason[r["reason"]] += r["signals"]

    meta = {
        "artifact": CLASSIFICATION_ARTIFACT_PATH.name,
        "artifact_sha256": sha,
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": date.today().isoformat(),
        "classified_asof": date.today().isoformat(),
        "corpus_window": CORPUS_WINDOW,
        "generator": "alpha/ml/generate_m4_security_type_artifact.py",
        "corpus_query": (
            "prior_artifact="
            f"{PRIOR_ARTIFACT_PATH.name}; hur_query="
            f"{' '.join(HUR_INCLUDED_QUERY.split())}"
        ),
        "resolution": (
            "FMP stable/profile (3 attempts, 2s backoff) -> "
            "fmp_delisted_companies name rules -> fail on unresolved"
        ),
        "semantics": {
            "ml_excluded": (
                "ticker security_type in NON_COMMON_TYPES "
                "(alpha.jobs.security_type); mark-don't-delete — "
                "signal_registry rows untouched, exclusion applied at "
                "training-set assembly via alpha.ml.security_type_exclusions"
            ),
            "row_coverage": (
                "union of the current artifact plus every HUR included ticker; "
                "zero unresolved/no_profile rows (generation fails otherwise)"
            ),
        },
        "prior_artifact": {
            "path": PRIOR_ARTIFACT_PATH.name,
            "metadata_path": PRIOR_METADATA_PATH.name,
            "tickers": len(prior_artifact_tickers),
        },
        "pit_caveat": (
            "Security type is TODAY'S (classified_asof) profile applied "
            "retroactively. De-SPACed/converted names now classify "
            "common_stock, so early-window SPAC contamination is "
            "understated; types are NOT point-in-time as of trading_date."
        ),
        "totals": {
            "corpus_signals": total_signals,
            "corpus_tickers": len(rows),
            "excluded_signals": excluded_signals,
            "excluded_tickers": len(excluded),
            "excluded_signal_pct": round(100 * excluded_signals / total_signals, 2),
        },
        "excluded_tickers_by_type": dict(
            Counter(r["security_type"] for r in excluded).most_common()
        ),
        "excluded_signals_by_type": dict(by_type.most_common()),
        "excluded_signals_by_reason": dict(by_reason.most_common()),
    }

    with open(CLASSIFICATION_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"wrote {CLASSIFICATION_ARTIFACT_PATH.name} sha256={sha}")
    print(f"totals: {meta['totals']}")
    return meta


if __name__ == "__main__":
    from alpha.runtime_env import load_runtime_env

    load_runtime_env()
    generate()
