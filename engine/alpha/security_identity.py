"""Shared helpers for consuming persisted security identity evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

from sqlalchemy.orm import Session

from alpha.db.models import SecurityIdentitySnapshot


@dataclass(frozen=True)
class ResolvedSecurityIdentity:
    """Canonical security identity for ML row-level de-duplication."""

    security_identity: str
    canonical_ticker: str


def load_security_identity_by_ticker(
    session: Session,
    scan_id: str,
) -> Dict[str, SecurityIdentitySnapshot]:
    """Load identity snapshots for a scan keyed by normalized ticker."""

    rows = (
        session.query(SecurityIdentitySnapshot)
        .filter(SecurityIdentitySnapshot.scan_id == scan_id)
        .all()
    )
    return {str(row.ticker).upper(): row for row in rows}


def _identity_key(identity: SecurityIdentitySnapshot | None, ticker: str) -> str:
    if identity is None:
        return f"ticker:{ticker.upper()}"
    for prefix, value in (
        ("share_class_figi", identity.share_class_figi),
        ("composite_figi", identity.composite_figi),
        ("cik", identity.cik),
        ("identity_hash", identity.identity_hash),
    ):
        if value:
            return f"{prefix}:{str(value).upper()}"
    return f"ticker:{str(identity.ticker or ticker).upper()}"


def _ticker_aliases_from_events(events_json: str | None) -> set[str]:
    if not events_json:
        return set()
    try:
        parsed = json.loads(events_json)
    except json.JSONDecodeError:
        return set()
    aliases: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if "ticker" in str(key).lower() and isinstance(child, str) and child:
                    aliases.add(child.upper())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(parsed)
    return aliases


def _old_ticker_aliases_from_events(events_json: str | None) -> set[str]:
    """Return aliases that explicitly represent prior ticker symbols."""

    if not events_json:
        return set()
    try:
        parsed = json.loads(events_json)
    except json.JSONDecodeError:
        return set()
    aliases: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).lower()
                if normalized_key in {
                    "old_ticker",
                    "previous_ticker",
                    "prior_ticker",
                    "from_ticker",
                } and isinstance(child, str) and child:
                    aliases.add(child.upper())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(parsed)
    return aliases


def _asof_sort_value(value: Any | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def identity_snapshot_sort_key(row: SecurityIdentitySnapshot) -> tuple[bool, datetime]:
    """Timezone-normalized PIT sort key for identity snapshot timestamps."""

    return (
        row.asof_timestamp is not None,
        _asof_sort_value(row.asof_timestamp),
    )


def load_security_identity_candidates_for_tickers(
    session: Session,
    tickers: Sequence[str],
    *,
    asof_timestamp: Any | None = None,
) -> list[SecurityIdentitySnapshot]:
    """Load identity rows whose direct ticker or persisted aliases match.

    Alias rows are loaded with one bounded broad query and then filtered by
    parsed event JSON in Python. This avoids constructing one leading-wildcard
    SQL predicate per requested ticker, which can exceed SQLite's expression
    depth and forces large non-sargable OR plans in PostgreSQL.
    """

    requested = sorted({str(ticker).upper() for ticker in tickers if ticker})
    if not requested:
        return []
    rows_by_id: dict[str, SecurityIdentitySnapshot] = {}

    query = session.query(SecurityIdentitySnapshot).filter(
        SecurityIdentitySnapshot.ticker.in_(requested),
    )
    if asof_timestamp is not None:
        query = query.filter(
            SecurityIdentitySnapshot.asof_timestamp.isnot(None),
            SecurityIdentitySnapshot.asof_timestamp <= asof_timestamp,
        )
    for row in query.all():
        rows_by_id[str(row.security_identity_snapshot_id)] = row

    alias_query = session.query(SecurityIdentitySnapshot).filter(
        SecurityIdentitySnapshot.ticker_events_json.isnot(None),
    )
    if asof_timestamp is not None:
        alias_query = alias_query.filter(
            SecurityIdentitySnapshot.asof_timestamp.isnot(None),
            SecurityIdentitySnapshot.asof_timestamp <= asof_timestamp,
        )
    requested_set = set(requested)
    for row in alias_query.all():
        if _ticker_aliases_from_events(row.ticker_events_json).intersection(requested_set):
            rows_by_id[str(row.security_identity_snapshot_id)] = row

    return list(rows_by_id.values())


def security_identity_snapshot_matches_ticker(
    row: SecurityIdentitySnapshot,
    ticker: str,
) -> bool:
    normalized = str(ticker or "").upper()
    if not normalized:
        return False
    if str(row.ticker or "").upper() == normalized:
        return True
    return normalized in _ticker_aliases_from_events(row.ticker_events_json)


def security_identity_snapshot_matches_prior_ticker_alias(
    row: SecurityIdentitySnapshot,
    ticker: str,
) -> bool:
    normalized = str(ticker or "").upper()
    return bool(normalized) and normalized in _old_ticker_aliases_from_events(
        row.ticker_events_json
    )


def _canonical_ticker(rows: Iterable[SecurityIdentitySnapshot]) -> str | None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.active is True,
            *identity_snapshot_sort_key(row),
            str(row.identity_hash or ""),
            str(row.ticker or ""),
        ),
        reverse=True,
    )
    return str(ordered[0].ticker).upper() if ordered else None


def resolve_security_identities_for_tickers(
    session: Session,
    tickers: Sequence[str],
) -> Dict[str, ResolvedSecurityIdentity]:
    """Resolve tickers to security identities, honoring persisted rename events.

    The ML trainer uses this as a defense-in-depth boundary: historical ticker
    renames and mergers should be one security for de-duplication, weighting,
    and CV purge purposes. Missing identity evidence intentionally falls back
    to the raw ticker, making absence visible but not fatal.
    """

    requested = {str(ticker).upper() for ticker in tickers if ticker}
    if not requested:
        return {}
    rows = load_security_identity_candidates_for_tickers(session, list(requested))
    groups: dict[str, list[SecurityIdentitySnapshot]] = {}
    direct_identity_by_ticker: dict[str, str] = {}
    alias_to_identity: dict[str, str] = {}

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row.ticker or "").upper(),
            row.active is True,
            *identity_snapshot_sort_key(row),
            str(row.identity_hash or ""),
            str(row.security_identity_snapshot_id or ""),
        ),
        reverse=True,
    )
    for row in ordered_rows:
        ticker = str(row.ticker or "").upper()
        key = _identity_key(row, ticker)
        groups.setdefault(key, []).append(row)
        if ticker:
            direct_identity_by_ticker.setdefault(ticker, key)
    alias_candidates_by_ticker: dict[str, set[str]] = {}
    for row in ordered_rows:
        ticker = str(row.ticker or "").upper()
        key = _identity_key(row, ticker)
        aliases = _old_ticker_aliases_from_events(row.ticker_events_json)
        for alias in aliases:
            if alias and alias in requested and alias not in direct_identity_by_ticker:
                alias_candidates_by_ticker.setdefault(alias, set()).add(key)
    for alias, keys in alias_candidates_by_ticker.items():
        if len(keys) == 1:
            alias_to_identity[alias] = next(iter(keys))
    canonical_by_identity = {
        key: _canonical_ticker(group) or key.rsplit(":", 1)[-1]
        for key, group in groups.items()
    }
    resolved: dict[str, ResolvedSecurityIdentity] = {}
    for ticker in requested:
        key = direct_identity_by_ticker.get(ticker) or alias_to_identity.get(ticker)
        if key is None:
            key = f"ticker:{ticker}"
            canonical = ticker
        else:
            canonical = canonical_by_identity.get(key, ticker)
        resolved[ticker] = ResolvedSecurityIdentity(
            security_identity=key,
            canonical_ticker=canonical,
        )
    return resolved


def security_identity_payload(
    identity: SecurityIdentitySnapshot | None,
) -> Dict[str, Any]:
    """Return feature-safe identity evidence without scan/job/database ids."""

    if identity is None:
        return {
            "identity_status": "unavailable",
            "identity_reason": "security_identity_snapshot_absent",
            "source_provider": "Polygon",
        }
    return {
        "identity_status": identity.identity_status,
        "identity_reason": identity.identity_reason,
        "identity_hash": identity.identity_hash,
        "cik": identity.cik,
        "composite_figi": identity.composite_figi,
        "share_class_figi": identity.share_class_figi,
        "active": identity.active,
        "delisted_utc": identity.delisted_utc,
        "list_date": identity.list_date,
        "polygon_type": identity.polygon_type,
        "polygon_market": identity.polygon_market,
        "polygon_locale": identity.polygon_locale,
        "primary_exchange": identity.polygon_primary_exchange,
        "name": identity.polygon_name,
        "source_provider": identity.source_provider,
        "source_endpoint": identity.source_endpoint,
        "source_payload_hash": identity.raw_payload_hash,
    }


def security_identity_lineage_ids(
    identity: SecurityIdentitySnapshot | None,
) -> List[str]:
    """Return persisted lineage ids for identity evidence."""

    if identity is None:
        return []
    ids: List[str] = []
    if identity.data_lineage_ids:
        try:
            parsed = json.loads(identity.data_lineage_ids)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            ids.extend(str(value) for value in parsed if value)
    for value in (identity.data_lineage_id, identity.events_data_lineage_id):
        if value and value not in ids:
            ids.append(value)
    return ids
