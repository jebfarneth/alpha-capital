"""Generate the M4 corpus security-type classification artifact (read-only).

Produces alpha/ml/data/m4_corpus_security_types_v7.csv and its sidecar
metadata for consumption by alpha.ml.security_type_exclusions. The corpus
database is only ever read (signal counts per ticker/month and the
fmp_delisted_companies directory); classification inputs come from the live
FMP profile endpoint. The only writes are the two artifact files.

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
    "pattern_id": "M4",
    "trading_date_min": "2024-01-01",
    "trading_date_max": "2026-06-30",
}
CORPUS_QUERY = """select left(trading_date,7) as month, ticker, count(*)
    from signal_registry
    where pattern_id = :pattern_id
      and trading_date between :trading_date_min and :trading_date_max
    group by 1, 2"""
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


def generate() -> dict:
    api_key = os.environ["FMP_API_KEY"]
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        db_rows = conn.execute(
            text(CORPUS_QUERY), CORPUS_WINDOW
        ).fetchall()
        counts: Counter = Counter()
        month_totals: Counter = Counter()
        ticker_months: Dict[str, Counter] = {}
        for month, ticker, n in db_rows:
            counts[ticker] += n
            month_totals[month] += n
            ticker_months.setdefault(ticker, Counter())[month] += n
        tickers = sorted(counts)
        print(f"corpus tickers: {len(tickers)}, signals: {sum(counts.values())}")

        with ThreadPoolExecutor(max_workers=10) as ex:
            profiles = dict(
                ex.map(lambda t: (t, _fetch_profile(t, api_key)), tickers)
            )

        classified: Dict[str, Tuple[str, str]] = {}
        missing = []
        for ticker in tickers:
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
    excluded_set = {r["ticker"] for r in excluded}
    month_excluded: Counter = Counter()
    for ticker in excluded_set:
        for month, n in ticker_months[ticker].items():
            month_excluded[month] += n
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
        "corpus_query": " ".join(CORPUS_QUERY.split()),
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
                "every ticker with >=1 M4 signal in corpus_window; zero "
                "unresolved/no_profile rows (generation fails otherwise)"
            ),
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
        "excluded_signals_by_month": {
            m: {"excluded": month_excluded[m], "total": month_totals[m]}
            for m in sorted(month_totals)
        },
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
