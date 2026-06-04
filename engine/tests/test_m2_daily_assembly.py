from __future__ import annotations

from datetime import date, datetime, timezone

from alpha.assembly.m2_daily import (
    CLASSIFICATION_OPPORTUNISTIC,
    CLASSIFICATION_UNCLASSIFIABLE,
    M2TransactionEvidence,
    assemble_m2_daily,
    classify_cmp_insider,
    first_tradable_session_after_publication,
    resolve_insider_identity,
    trade_size_weights,
)
from alpha.jobs.detector_orchestration import _input_asof_ceiling


def _ts(day: int = 3, hour: int = 22) -> datetime:
    return datetime(2026, 6, day, hour, 0, tzinfo=timezone.utc)


def _snapshot(ticker: str = "ACME") -> dict:
    return {
        "ticker": ticker,
        "universe_snapshot_id": f"snap-{ticker}",
        "asof_timestamp": datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc),
        "market_cap": 75_000_000,
        "price": 5.25,
        "primary_exchange": "NASDAQ",
        "security_type": "common_stock",
        "operating_universe_inclusion": True,
        "source_lineage_hash": "universe-hash",
    }


def _tx(
    insider: str,
    *,
    tx_date: str,
    accepted_at: datetime,
    first_tradable: str,
    accession: str,
    code: str = "P",
    open_purchase: bool = True,
    market_cap: float = 75_000_000,
) -> M2TransactionEvidence:
    return M2TransactionEvidence(
        transaction_id=f"{accession}:{insider}:{tx_date}:{code}",
        ticker="ACME",
        source_authority="sec_edgar",
        insider_id=f"cik:{insider}",
        insider_cik=insider,
        insider_name=f"Owner {insider}",
        identity_resolution_method="sec_reporting_owner_cik",
        identity_resolution_confidence=1.0,
        filing_accession_number=accession,
        filing_form="4",
        filing_date=accepted_at.date().isoformat(),
        filing_accepted_at=accepted_at,
        filing_detected_at=accepted_at,
        first_tradable_session=first_tradable,
        clock_quality="accepted_detected",
        transaction_date=tx_date,
        transaction_code=code,
        acquired_disposed_code="A" if code == "P" else "D",
        transaction_shares=10_000,
        transaction_price_per_share=2.0,
        purchase_notional_usd=20_000 if open_purchase else None,
        market_cap_usd=market_cap,
        issuer_state="CA",
        insider_state="CA",
        is_open_market_purchase=open_purchase,
        is_buy=open_purchase,
        is_sell=not open_purchase,
        data_lineage_ids=["lin"],
        lineage_hashes=["hash"],
    )


def _history(insider: str) -> list[M2TransactionEvidence]:
    return [
        _tx(insider, tx_date="2023-01-15", accepted_at=datetime(2023, 1, 16, 17, tzinfo=timezone.utc), first_tradable="2023-01-17", accession=f"{insider}-23-000001", open_purchase=False, code="S"),
        _tx(insider, tx_date="2024-02-15", accepted_at=datetime(2024, 2, 16, 17, tzinfo=timezone.utc), first_tradable="2024-02-20", accession=f"{insider}-24-000001", open_purchase=False, code="S"),
        _tx(insider, tx_date="2025-03-15", accepted_at=datetime(2025, 3, 17, 17, tzinfo=timezone.utc), first_tradable="2025-03-18", accession=f"{insider}-25-000001", open_purchase=False, code="S"),
    ]


def test_after_close_form4_anchors_to_next_regular_open():
    clock = first_tradable_session_after_publication(
        filing_accepted_at=datetime(2026, 6, 3, 22, 0, tzinfo=timezone.utc),
        filing_detected_at=datetime(2026, 6, 3, 22, 5, tzinfo=timezone.utc),
        filing_date=date(2026, 6, 3),
    )

    assert clock.first_tradable_session == date(2026, 6, 4)
    assert clock.clock_quality == "accepted_detected"


def test_identity_resolver_collapses_name_variants_when_sec_cik_matches():
    left = resolve_insider_identity(sec_owner_cik="12345", owner_name="Jane Q. Doe")
    right = resolve_insider_identity(sec_owner_cik="0000012345", owner_name="JANE DOE")

    assert left.insider_id == right.insider_id == "cik:0000012345"
    assert left.cik_backed is True


def test_cmp_classifier_uses_jan1_pit_cutoff_not_same_year_acceptance():
    txs = _history("0000000001")
    txs.append(_tx(
        "0000000001",
        tx_date="2025-12-29",
        accepted_at=datetime(2026, 1, 3, 15, tzinfo=timezone.utc),
        first_tradable="2026-01-05",
        accession="late-filed-yminus1",
        open_purchase=False,
        code="S",
    ))

    classification = classify_cmp_insider(
        txs,
        insider_id="cik:0000000001",
        calendar_year=2026,
    )

    assert classification.classification == CLASSIFICATION_OPPORTUNISTIC
    assert "2025" in classification.basis["months_by_year"]
    assert 12 not in classification.basis["months_by_year"]["2025"]


def test_market_cap_relative_size_weight_preserves_microcap_scale():
    micro_weights = trade_size_weights(
        purchase_notional_usd=50_000,
        market_cap_usd=30_000_000,
        prior_purchase_notionals=[10_000, 15_000, 20_000],
    )
    upper_band_weights = trade_size_weights(
        purchase_notional_usd=50_000,
        market_cap_usd=250_000_000,
    )

    assert micro_weights is not None
    assert upper_band_weights is not None
    assert micro_weights.used_market_cap_relative is True
    assert micro_weights.production_weight > upper_band_weights.production_weight
    assert micro_weights.shadow_own_history is not None


def test_assemble_m2_cluster_uses_accessions_cik_count_and_next_open_age_zero():
    transactions = []
    for insider, accession in (
        ("0000000001", "0000000001-26-000001"),
        ("0000000002", "0000000002-26-000001"),
    ):
        transactions.extend(_history(insider))
        transactions.append(_tx(
            insider,
            tx_date="2026-06-03",
            accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
            first_tradable="2026-06-04",
            accession=accession,
        ))

    assembly = assemble_m2_daily(
        snapshots=[_snapshot()],
        transactions=transactions,
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc),
        decision_date="2026-06-03",
        evidence_session_date="2026-06-03",
        next_execution_session="2026-06-04",
    )

    assert assembly["M2"].assembled_count == 1
    payload = assembly["M2"].inputs[0].market_data
    assert payload["n_distinct_opp_buyers_30d"] == 2
    assert payload["days_since_last_opp_buy_filing_detected"] == 0
    assert payload["sec_accession_numbers"] == [
        "0000000001-26-000001",
        "0000000002-26-000001",
    ]
    assert len(payload["m2_cluster_members"]) == 2


def test_assemble_m2u_shadow_uses_own_pattern_id_for_unclassifiable_cluster():
    transactions = [
        _tx(
            "0000000003",
            tx_date="2026-06-03",
            accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
            first_tradable="2026-06-04",
            accession="0000000003-26-000001",
        ),
        _tx(
            "0000000004",
            tx_date="2026-06-03",
            accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
            first_tradable="2026-06-04",
            accession="0000000004-26-000001",
        ),
    ]

    assert classify_cmp_insider(
        transactions,
        insider_id="cik:0000000003",
        calendar_year=2026,
    ).classification == CLASSIFICATION_UNCLASSIFIABLE
    assembly = assemble_m2_daily(
        snapshots=[_snapshot()],
        transactions=transactions,
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc),
        decision_date="2026-06-03",
        evidence_session_date="2026-06-03",
        next_execution_session="2026-06-04",
    )

    assert assembly["M2"].assembled_count == 0
    assert assembly["M2U"].assembled_count == 1
    assert assembly["M2U"].inputs[0].market_data["shadow_pattern_id"] == "M2U"


def test_orchestration_asof_ceiling_allows_after_close_same_decision_day():
    inp = assemble_m2_daily(
        snapshots=[_snapshot()],
        transactions=[
            *_history("0000000001"),
            *_history("0000000002"),
            _tx("0000000001", tx_date="2026-06-03", accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc), first_tradable="2026-06-04", accession="0000000001-26-000001"),
            _tx("0000000002", tx_date="2026-06-03", accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc), first_tradable="2026-06-04", accession="0000000002-26-000001"),
        ],
        cutoff_timestamp=_ts(),
        universe_cutoff_timestamp=datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc),
        decision_date="2026-06-03",
        evidence_session_date="2026-06-03",
        next_execution_session="2026-06-04",
    )["M2"].inputs[0]

    ceiling, label = _input_asof_ceiling(inp, None, "2026-06-03")

    assert label == "input asof ceiling"
    assert ceiling == _ts()
