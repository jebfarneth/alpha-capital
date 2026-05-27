"""Shared helpers for consuming persisted security identity evidence."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from alpha.db.models import SecurityIdentitySnapshot


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
