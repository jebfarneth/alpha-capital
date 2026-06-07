"""Production M2 insider-cluster daily feature assembly wiring."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.m2_daily import (
    M2UDetector,
    assemble_m2_daily,
    classify_cmp_insider,
    cluster_member_rows,
    first_tradable_session_after_publication,
    transaction_evidence_from_fmp,
    transaction_evidence_from_sec,
)
from alpha.data.contracts import stable_hash
from alpha.data.edgar import FORM4_TRANSACTIONS_ENDPOINT
from alpha.data.fmp import INSIDER_TRADING_SEARCH_ENDPOINT
from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJobRun,
    M2ClusterMember,
    M2InsiderClassification,
    M2InsiderTransaction,
    M2SecFetchCoverage,
    SecurityIdentitySnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.universe_builder import market_cap_bucket_counts
from alpha.market_calendar import EASTERN_TZ, resolve_us_equity_session
from alpha.patterns.m2 import M2Detector


class M2DailyAssemblyJob(BaseJob):
    """Run daily M2 from canonical universe through persisted orchestration."""

    job_name = "m2_daily_feature_assembly"
    job_type = "feature_assembly"

    def __init__(
        self,
        session: Session,
        *,
        sec_adapter: Any,
        fmp_adapter: Optional[Any] = None,
        run_timestamp: Optional[datetime] = None,
        form4_lookback_calendar_days: int = 1465,
        fmp_page_limit: int = 100,
        skip_fmp_enrichment: bool = False,
    ):
        self._session = session
        self._sec_adapter = sec_adapter
        self._fmp_adapter = fmp_adapter
        self._run_timestamp = run_timestamp
        self._form4_lookback_calendar_days = form4_lookback_calendar_days
        self._fmp_page_limit = fmp_page_limit
        self._skip_fmp_enrichment = skip_fmp_enrichment

    def run(self, ctx: JobContext) -> JobResult:
        run_timestamp, timestamp_error = _resolve_run_timestamp(
            self._run_timestamp,
            ctx.params.get("run_timestamp"),
            ctx.started_at,
        )
        if timestamp_error:
            return JobResult(
                status="failed",
                errors=[{"stage": "params", "message": timestamp_error}],
            )

        session_resolution = resolve_us_equity_session(run_timestamp)
        decision_date = session_resolution.decision_date
        evidence_session_date = session_resolution.evidence_session_date
        decision_day = date.fromisoformat(decision_date)
        history_from_date = decision_day - timedelta(days=self._form4_lookback_calendar_days)

        requested_trading_date = ctx.params.get("trading_date")
        if requested_trading_date and requested_trading_date != decision_date:
            return JobResult(
                status="failed",
                metrics={"decision_date": decision_date},
                errors=[{
                    "stage": "params",
                    "message": (
                        "trading_date must match resolver decision_date; "
                        f"got {requested_trading_date}, resolved {decision_date}"
                    ),
                }],
            )

        scan_id, scan_asof_timestamp, snapshots, canonical_error = (
            _load_included_canonical_snapshots(self._session, decision_date)
        )
        if canonical_error:
            return JobResult(
                status="failed",
                metrics=_base_metrics(
                    session_resolution,
                    scan_id=scan_id,
                    included_universe_size=0,
                    from_date=history_from_date,
                    to_date=decision_day,
                ),
                errors=[{"stage": "canonical_universe", "message": canonical_error}],
            )

        included_market_cap_bucket_counts = market_cap_bucket_counts(snapshots)
        identity_by_ticker = _load_identity_ciks(self._session, scan_id or "")
        lineage_ids_by_ticker: Dict[str, List[str]] = {}
        lineage_hashes_by_ticker: Dict[str, List[str]] = {}
        fetch_errors: List[Dict[str, Any]] = []
        sec_transaction_count = 0
        fmp_enrichment_count = 0
        unresolved_cik_count = 0
        live_detection_window_start, has_prior_detection_window = _live_detection_window_start(
            self._session,
            job_id=ctx.job_id,
            current_job_run_id=ctx.job_run_id,
            current_started_at=ctx.started_at,
            evidence_session_date=evidence_session_date,
        )
        sec_fetch_from_date = _sec_form4_fetch_from_date(
            history_from_date=history_from_date,
            live_detection_window_start=live_detection_window_start,
            has_prior_detection_window=has_prior_detection_window,
        )
        tickers = [snapshot.ticker.upper() for snapshot in snapshots]
        tickers_with_transactions = _tickers_with_transaction_history(
            self._session,
            tickers=tickers,
            from_date=history_from_date,
        )
        tickers_with_fetch_coverage = _tickers_with_sec_fetch_coverage(
            self._session,
            tickers=tickers,
            ticker_to_cik=identity_by_ticker,
            from_date=history_from_date,
        )
        tickers_with_history = tickers_with_transactions | tickers_with_fetch_coverage
        full_history_fetch_tickers: List[str] = []

        for snapshot in snapshots:
            ticker = snapshot.ticker.upper()
            ticker_fetch_from_date = (
                sec_fetch_from_date
                if ticker in tickers_with_history
                else history_from_date
            )
            if ticker_fetch_from_date == history_from_date:
                full_history_fetch_tickers.append(ticker)
            issuer_cik = identity_by_ticker.get(ticker)
            if not issuer_cik:
                ticker_resp = self._sec_adapter.get_company_ticker(
                    ticker,
                    asof=run_timestamp,
                )
                lineage = _record_response_lineage(
                    self._session,
                    ticker_resp,
                    job_run_id=ctx.job_run_id,
                    raw_payload={
                        "endpoint": "sec_company_ticker",
                        "ticker": ticker,
                        "row": _jsonable(ticker_resp.data),
                    },
                )
                lineage_ids_by_ticker.setdefault(ticker, []).append(lineage.data_lineage_id)
                lineage_hashes_by_ticker.setdefault(ticker, []).append(
                    ticker_resp.lineage.raw_payload_hash
                )
                if ticker_resp.ok and ticker_resp.data is not None:
                    issuer_cik = ticker_resp.data.cik_str
                else:
                    unresolved_cik_count += 1
                    if not ticker_resp.ok:
                        fetch_errors.append({"ticker": ticker, "stage": "sec_ticker", **_provider_error(ticker_resp.error)})
                    continue

            sec_resp = self._sec_adapter.get_form4_transactions(
                issuer_cik,
                from_date=ticker_fetch_from_date,
                to_date=decision_day,
                asof=run_timestamp,
            )
            sec_lineage = _record_response_lineage(
                self._session,
                sec_resp,
                job_run_id=ctx.job_run_id,
                raw_payload={
                    "endpoint": FORM4_TRANSACTIONS_ENDPOINT,
                    "ticker": ticker,
                    "issuer_cik": issuer_cik,
                    "from": ticker_fetch_from_date.isoformat(),
                    "to": decision_day.isoformat(),
                    "transaction_ids": [
                        getattr(row, "transaction_id", None)
                        for row in (sec_resp.data or [])
                    ],
                },
            )
            lineage_ids_by_ticker.setdefault(ticker, []).append(sec_lineage.data_lineage_id)
            lineage_hashes_by_ticker.setdefault(ticker, []).append(sec_resp.lineage.raw_payload_hash)
            if not sec_resp.ok:
                fetch_errors.append({"ticker": ticker, "stage": "sec_form4", **_provider_error(sec_resp.error)})
                continue
            sec_evidence = [
                transaction_evidence_from_sec(
                    row,
                    detected_at=_live_detected_at_for_sec_row(
                        row,
                        run_timestamp=run_timestamp,
                        live_detection_window_start=live_detection_window_start,
                    ),
                    market_cap_usd=snapshot.market_cap,
                    ticker=ticker,
                    lineage_ids=[sec_lineage.data_lineage_id],
                    lineage_hashes=[sec_resp.lineage.raw_payload_hash],
                )
                for row in (sec_resp.data or [])
            ]
            _persist_sec_fetch_coverage(
                self._session,
                ticker=ticker,
                issuer_cik=issuer_cik,
                from_date=ticker_fetch_from_date,
                to_date=decision_day,
                transaction_count=len(sec_evidence),
                scan_id=scan_id,
                snapshot=snapshot,
                job_run_id=ctx.job_run_id,
                data_lineage_id=sec_lineage.data_lineage_id,
                raw_payload_hash=sec_resp.lineage.raw_payload_hash,
            )
            for evidence in sec_evidence:
                _persist_transaction(
                    self._session,
                    evidence,
                    scan_id=scan_id,
                    snapshot=snapshot,
                    job_run_id=ctx.job_run_id,
                )
            sec_transaction_count += len(sec_evidence)

            if self._skip_fmp_enrichment or self._fmp_adapter is None:
                continue
            fmp_resp = self._fmp_adapter.get_insider_trades(
                symbol=ticker,
                page=0,
                limit=self._fmp_page_limit,
                asof=run_timestamp,
            )
            fmp_lineage = _record_response_lineage(
                self._session,
                fmp_resp,
                job_run_id=ctx.job_run_id,
                raw_payload={
                    "endpoint": INSIDER_TRADING_SEARCH_ENDPOINT,
                    "ticker": ticker,
                    "page": 0,
                    "limit": self._fmp_page_limit,
                    "accessions": [
                        getattr(row, "accession_number", None)
                        for row in (fmp_resp.data or [])
                    ],
                },
            )
            lineage_ids_by_ticker[ticker].append(fmp_lineage.data_lineage_id)
            lineage_hashes_by_ticker[ticker].append(fmp_resp.lineage.raw_payload_hash)
            if not fmp_resp.ok:
                fetch_errors.append({"ticker": ticker, "stage": "fmp_insider", **_provider_error(fmp_resp.error)})
                continue
            accession_owner_ciks = _accession_owner_ciks(sec_evidence)
            existing_keys = {_dedupe_key(evidence) for evidence in sec_evidence}
            for trade in fmp_resp.data or []:
                accession = getattr(trade, "accession_number", None)
                if not accession:
                    continue
                evidence = transaction_evidence_from_fmp(
                    trade,
                    accession_owner_cik=accession_owner_ciks.get(accession),
                    detected_at=None,
                    market_cap_usd=snapshot.market_cap,
                    lineage_ids=[fmp_lineage.data_lineage_id, sec_lineage.data_lineage_id],
                    lineage_hashes=[
                        fmp_resp.lineage.raw_payload_hash,
                        sec_resp.lineage.raw_payload_hash,
                    ],
                )
                if evidence is None or _dedupe_key(evidence) in existing_keys:
                    continue
                _persist_transaction(
                    self._session,
                    evidence,
                    scan_id=scan_id,
                    snapshot=snapshot,
                    job_run_id=ctx.job_run_id,
                )
                fmp_enrichment_count += 1

        transactions = _load_transactions(
            self._session,
            tickers=tickers,
            from_date=history_from_date,
        )
        _persist_classifications(
            self._session,
            transactions,
            calendar_year=date.fromisoformat(session_resolution.next_execution_session).year,
        )
        assembly = assemble_m2_daily(
            snapshots=snapshots,
            transactions=transactions,
            cutoff_timestamp=run_timestamp,
            universe_cutoff_timestamp=scan_asof_timestamp,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=session_resolution.next_execution_session,
            lineage_ids_by_ticker=lineage_ids_by_ticker,
            lineage_hashes_by_ticker=lineage_hashes_by_ticker,
        )
        for pattern_id, assembly_result in assembly.items():
            _persist_cluster_members(
                self._session,
                assembly_result.inputs,
                pattern_id=pattern_id,
            )

        orchestration = DetectorOrchestrationJob(
            self._session,
            detectors=[M2Detector(), M2UDetector()],
            trading_date=decision_date,
            assembled_inputs={
                "M2": assembly["M2"].inputs,
                "M2U": assembly["M2U"].inputs,
            },
        )
        orchestration_result = orchestration.run(ctx)

        metrics = _base_metrics(
            session_resolution,
            scan_id=scan_id,
            included_universe_size=len(snapshots),
            from_date=history_from_date,
            to_date=decision_day,
            included_market_cap_bucket_counts=included_market_cap_bucket_counts,
        )
        metrics.update({
            "sec_form4_fetch_from_date": sec_fetch_from_date.isoformat(),
            "prior_detection_window_start": live_detection_window_start.isoformat(),
            "prior_detection_window_found": has_prior_detection_window,
            # Widening the canonical universe can cold-pull M2 until newly
            # included names have transaction history or successful SEC fetch
            # coverage, including no-Form-4 coverage.
            "tickers_with_transaction_history_count": len(tickers_with_transactions),
            "tickers_with_transaction_history_sample": sorted(tickers_with_transactions)[:50],
            "tickers_with_sec_fetch_coverage_count": len(tickers_with_fetch_coverage),
            "tickers_with_sec_fetch_coverage_sample": sorted(tickers_with_fetch_coverage)[:50],
            "tickers_with_m2_warm_coverage_count": len(tickers_with_history),
            "tickers_with_m2_warm_coverage_sample": sorted(tickers_with_history)[:50],
            "full_history_fetch_ticker_count": len(full_history_fetch_tickers),
            "full_history_fetch_ticker_sample": sorted(full_history_fetch_tickers)[:50],
            "m2_warm_path_requires_seeded_history": bool(full_history_fetch_tickers),
            "sec_transaction_count": sec_transaction_count,
            "fmp_enrichment_count": fmp_enrichment_count,
            "unresolved_cik_count": unresolved_cik_count,
            "fetch_error_count": len(fetch_errors),
            "assembly": {
                pattern_id: _assembly_metrics(result)
                for pattern_id, result in assembly.items()
            },
            "orchestration": orchestration_result.metrics,
        })
        errors = list(orchestration_result.errors or [])
        errors.extend(fetch_errors)
        fired_tickers = _fired_m2_tickers(
            self._session,
            job_run_id=ctx.job_run_id,
        )
        fatal_fetch_errors, tolerated_fetch_errors = _partition_fetch_errors(
            fetch_errors,
            fired_tickers=fired_tickers,
        )
        metrics.update({
            "fatal_fetch_error_count": len(fatal_fetch_errors),
            "tolerated_fetch_error_count": len(tolerated_fetch_errors),
            "fatal_fetch_errors": _fetch_error_summaries(fatal_fetch_errors),
        })
        # A fetch error that prevents a would-be M2/M2U fire from entering the
        # fired set is tolerated by design. This mirrors canonical source gating:
        # non-fired source attempts do not fail the run, while a fire certified
        # from incomplete source evidence is fatal.
        status = (
            "partial_failed"
            if orchestration_result.status == "partial_failed" or fatal_fetch_errors
            else "finished"
        )
        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "scan_id": scan_id,
                "decision_date": decision_date,
                "from_date": history_from_date.isoformat(),
                "sec_fetch_from_date": sec_fetch_from_date.isoformat(),
                "to_date": decision_day.isoformat(),
            },
            output_hashes={
                "assembly": stable_hash(metrics["assembly"]),
                **(orchestration_result.output_hashes or {}),
            },
            errors=errors,
        )


def _resolve_run_timestamp(
    explicit: Optional[datetime],
    param_value: Any,
    fallback: datetime,
) -> Tuple[datetime, Optional[str]]:
    value = explicit
    if value is None and param_value:
        try:
            value = datetime.fromisoformat(str(param_value).replace("Z", "+00:00"))
        except ValueError:
            return fallback, f"invalid run_timestamp: {param_value}"
    if value is None:
        value = fallback
    if value.tzinfo is None or value.utcoffset() is None:
        return value, "run_timestamp must be timezone-aware"
    return value.astimezone(timezone.utc), None


def _load_included_canonical_snapshots(
    session: Session,
    trading_date: str,
) -> Tuple[Optional[str], Optional[datetime], List[UniverseSnapshot], Optional[str]]:
    canonical = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if canonical is None:
        return None, None, [], f"no canonical universe scan for trading_date={trading_date}"
    scan = session.get(UniverseScan, canonical.scan_id)
    if scan is None:
        return None, None, [], f"canonical scan_id {canonical.scan_id} not found"
    snapshots = (
        session.query(UniverseSnapshot)
        .filter(
            UniverseSnapshot.scan_id == canonical.scan_id,
            UniverseSnapshot.operating_universe_inclusion.is_(True),
        )
        .all()
    )
    return canonical.scan_id, _ensure_aware(scan.asof_timestamp), snapshots, None


def _load_identity_ciks(session: Session, scan_id: str) -> Dict[str, str]:
    rows = (
        session.query(SecurityIdentitySnapshot)
        .filter(SecurityIdentitySnapshot.scan_id == scan_id)
        .all()
    )
    return {
        row.ticker.upper(): row.cik
        for row in rows
        if row.cik
    }


def _live_detection_window_start(
    session: Session,
    *,
    job_id: str,
    current_job_run_id: str,
    current_started_at: datetime,
    evidence_session_date: str,
) -> Tuple[datetime, bool]:
    previous_run = (
        session.query(EvidenceJobRun)
        .filter(
            EvidenceJobRun.job_id == job_id,
            EvidenceJobRun.job_run_id != current_job_run_id,
            EvidenceJobRun.run_status == "finished",
            EvidenceJobRun.started_at.isnot(None),
            EvidenceJobRun.started_at < current_started_at,
            EvidenceJobRun.ended_at.isnot(None),
        )
        .order_by(EvidenceJobRun.started_at.desc())
        .first()
    )
    if previous_run is not None:
        previous_started_at = _ensure_aware(previous_run.started_at)
        if previous_started_at is not None:
            return previous_started_at, True
    evidence_day = date.fromisoformat(evidence_session_date)
    return (
        datetime.combine(evidence_day, time.min, EASTERN_TZ).astimezone(timezone.utc),
        False,
    )


def _sec_form4_fetch_from_date(
    *,
    history_from_date: date,
    live_detection_window_start: datetime,
    has_prior_detection_window: bool,
) -> date:
    if not has_prior_detection_window:
        return history_from_date
    incremental_start = live_detection_window_start.astimezone(EASTERN_TZ).date()
    return max(history_from_date, incremental_start)


def _live_detected_at_for_sec_row(
    row: Any,
    *,
    run_timestamp: datetime,
    live_detection_window_start: datetime,
) -> Optional[datetime]:
    accepted_at = _ensure_aware(getattr(row, "filing_accepted_at", None))
    if accepted_at is None:
        return None
    if live_detection_window_start <= accepted_at <= run_timestamp:
        return run_timestamp
    return accepted_at


def _persist_transaction(
    session: Session,
    evidence: Any,
    *,
    scan_id: Optional[str],
    snapshot: UniverseSnapshot,
    job_run_id: Optional[str],
) -> None:
    existing = session.get(M2InsiderTransaction, evidence.transaction_id)
    persisted_detected_at = evidence.filing_detected_at
    if existing is not None and existing.filing_detected_at is not None:
        persisted_detected_at = _ensure_aware(existing.filing_detected_at)
    if persisted_detected_at != evidence.filing_detected_at:
        evidence = _with_detection_clock(evidence, persisted_detected_at)
    if existing is not None:
        session.delete(existing)
        session.flush()
    row = M2InsiderTransaction(
        transaction_id=evidence.transaction_id,
        scan_id=scan_id,
        universe_snapshot_id=snapshot.universe_snapshot_id,
        job_run_id=job_run_id,
        source_authority=evidence.source_authority,
        enrichment_sources=json.dumps(["SEC_EDGAR"] if evidence.source_authority == "sec_edgar" else ["FMP", "SEC_EDGAR"]),
        ticker=evidence.ticker,
        issuer_cik=evidence.issuer_cik,
        issuer_name=evidence.issuer_name,
        insider_id=evidence.insider_id,
        insider_cik=evidence.insider_cik,
        insider_name=evidence.insider_name,
        issuer_state=evidence.issuer_state,
        insider_state=evidence.insider_state,
        identity_resolution_method=evidence.identity_resolution_method,
        identity_resolution_confidence=evidence.identity_resolution_confidence,
        filing_accession_number=evidence.filing_accession_number,
        filing_form=evidence.filing_form,
        filing_date=evidence.filing_date,
        filing_accepted_at=evidence.filing_accepted_at,
        filing_detected_at=persisted_detected_at,
        first_tradable_session=evidence.first_tradable_session,
        clock_quality=evidence.clock_quality,
        transaction_date=evidence.transaction_date,
        transaction_code=evidence.transaction_code,
        acquired_disposed_code=evidence.acquired_disposed_code,
        transaction_shares=evidence.transaction_shares,
        transaction_price_per_share=evidence.transaction_price_per_share,
        transaction_notional_usd=(
            abs(evidence.transaction_shares * evidence.transaction_price_per_share)
            if evidence.transaction_shares is not None
            and evidence.transaction_price_per_share is not None
            else None
        ),
        purchase_notional_usd=evidence.purchase_notional_usd,
        market_cap_usd=evidence.market_cap_usd,
        ownership_type=evidence.ownership_type,
        insider_roles_json=json.dumps(evidence.insider_roles, default=str),
        is_open_market_purchase=evidence.is_open_market_purchase,
        is_buy=evidence.is_buy,
        is_sell=evidence.is_sell,
        is_10b5_1=evidence.is_10b5_1,
        sec_fmp_mismatch=evidence.sec_fmp_mismatch,
        data_lineage_ids=json.dumps(evidence.data_lineage_ids),
        raw_json=json.dumps(evidence.raw, default=str),
    )
    session.add(row)
    session.flush()


def _persist_sec_fetch_coverage(
    session: Session,
    *,
    ticker: str,
    issuer_cik: str,
    from_date: date,
    to_date: date,
    transaction_count: int,
    scan_id: Optional[str],
    snapshot: UniverseSnapshot,
    job_run_id: str,
    data_lineage_id: Optional[str],
    raw_payload_hash: Optional[str],
) -> None:
    ticker = ticker.upper()
    from_value = from_date.isoformat()
    to_value = to_date.isoformat()
    existing = (
        session.query(M2SecFetchCoverage)
        .filter(
            M2SecFetchCoverage.ticker == ticker,
            M2SecFetchCoverage.issuer_cik == issuer_cik,
            M2SecFetchCoverage.from_date == from_value,
        )
        .first()
    )
    if existing is None:
        existing = M2SecFetchCoverage(
            ticker=ticker,
            issuer_cik=issuer_cik,
            from_date=from_value,
        )
        session.add(existing)
    existing.to_date = max(str(existing.to_date or to_value), to_value)
    existing.status = "success"
    existing.transaction_count = transaction_count
    existing.scan_id = scan_id
    existing.universe_snapshot_id = snapshot.universe_snapshot_id
    existing.job_run_id = job_run_id
    existing.data_lineage_id = data_lineage_id
    existing.raw_payload_hash = raw_payload_hash
    existing.fetched_at = datetime.now(timezone.utc)
    session.flush()


def _load_transactions(
    session: Session,
    *,
    tickers: List[str],
    from_date: date,
) -> List[M2InsiderTransaction]:
    return (
        session.query(M2InsiderTransaction)
        .filter(
            M2InsiderTransaction.ticker.in_(tickers),
            M2InsiderTransaction.transaction_date >= from_date.isoformat(),
        )
        .all()
    )


def _tickers_with_transaction_history(
    session: Session,
    *,
    tickers: List[str],
    from_date: date,
) -> set[str]:
    if not tickers:
        return set()
    rows = (
        session.query(M2InsiderTransaction.ticker)
        .filter(
            M2InsiderTransaction.ticker.in_(tickers),
            M2InsiderTransaction.transaction_date >= from_date.isoformat(),
        )
        .distinct()
        .all()
    )
    return {str(row[0]).upper() for row in rows if row[0]}


def _tickers_with_sec_fetch_coverage(
    session: Session,
    *,
    tickers: List[str],
    ticker_to_cik: Dict[str, str],
    from_date: date,
) -> set[str]:
    if not tickers:
        return set()
    rows = (
        session.query(
            M2SecFetchCoverage.ticker,
            M2SecFetchCoverage.issuer_cik,
        )
        .filter(
            M2SecFetchCoverage.ticker.in_(tickers),
            M2SecFetchCoverage.from_date <= from_date.isoformat(),
            M2SecFetchCoverage.status == "success",
        )
        .distinct()
        .all()
    )
    covered: set[str] = set()
    for ticker, issuer_cik in rows:
        normalized = str(ticker or "").upper()
        if not normalized:
            continue
        expected_cik = ticker_to_cik.get(normalized)
        if expected_cik and issuer_cik and str(issuer_cik) != str(expected_cik):
            continue
        covered.add(normalized)
    return covered


def _with_detection_clock(evidence: Any, detected_at: Optional[datetime]) -> Any:
    filing_day = None
    if evidence.filing_date:
        try:
            filing_day = date.fromisoformat(str(evidence.filing_date)[:10])
        except ValueError:
            filing_day = None
    clock = first_tradable_session_after_publication(
        filing_accepted_at=evidence.filing_accepted_at,
        filing_detected_at=detected_at,
        filing_date=filing_day,
    )
    return replace(
        evidence,
        filing_detected_at=detected_at,
        first_tradable_session=clock.first_tradable_session.isoformat(),
        clock_quality=clock.clock_quality,
    )


def _persist_classifications(
    session: Session,
    transactions: List[M2InsiderTransaction],
    *,
    calendar_year: int,
) -> None:
    transactions_by_insider = _transactions_by_insider(transactions)
    for insider_id, insider_transactions in sorted(transactions_by_insider.items()):
        classification = classify_cmp_insider(
            insider_transactions,
            insider_id=insider_id,
            calendar_year=calendar_year,
        )
        existing = (
            session.query(M2InsiderClassification)
            .filter(
                M2InsiderClassification.insider_id == insider_id,
                M2InsiderClassification.calendar_year == calendar_year,
            )
            .first()
        )
        if existing is not None:
            session.delete(existing)
            session.flush()
        sample = insider_transactions[0] if insider_transactions else None
        session.add(M2InsiderClassification(
            insider_id=insider_id,
            insider_cik=getattr(sample, "insider_cik", None),
            insider_name=getattr(sample, "insider_name", None),
            calendar_year=calendar_year,
            classification=classification.classification,
            routine_month=classification.routine_month,
            prior_year_count=classification.prior_year_count,
            data_cutoff_at=classification.data_cutoff_at,
            basis_json=json.dumps(classification.basis, default=str),
        ))
    session.flush()


def _transactions_by_insider(
    transactions: List[M2InsiderTransaction],
) -> Dict[str, List[M2InsiderTransaction]]:
    by_insider: Dict[str, List[M2InsiderTransaction]] = {}
    for row in transactions:
        if not row.insider_id:
            continue
        by_insider.setdefault(row.insider_id, []).append(row)
    return by_insider


def _persist_cluster_members(
    session: Session,
    inputs: List[Any],
    *,
    pattern_id: str,
) -> None:
    for inp in inputs:
        cluster_id = inp.market_data.get("m2_cluster_id")
        if not cluster_id:
            continue
        (
            session.query(M2ClusterMember)
            .filter(
                M2ClusterMember.pattern_id == pattern_id,
                M2ClusterMember.m2_cluster_id == cluster_id,
            )
            .delete(synchronize_session=False)
        )
        for payload in cluster_member_rows(inp, pattern_id):
            if not payload.get("transaction_id"):
                continue
            session.add(M2ClusterMember(**payload))
    session.flush()


def _fired_m2_tickers(session: Session, *, job_run_id: str) -> set[str]:
    rows = (
        session.query(SignalRegistry.ticker)
        .filter(
            SignalRegistry.job_run_id == job_run_id,
            SignalRegistry.pattern_id.in_(("M2", "M2U")),
        )
        .distinct()
        .all()
    )
    return {str(row[0]).strip().upper() for row in rows if str(row[0] or "").strip()}


def _partition_fetch_errors(
    fetch_errors: List[Dict[str, Any]],
    *,
    fired_tickers: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fatal: List[Dict[str, Any]] = []
    tolerated: List[Dict[str, Any]] = []
    normalized_fired = {str(ticker).strip().upper() for ticker in fired_tickers}
    for error in fetch_errors:
        ticker = str(error.get("ticker") or "").strip().upper()
        if ticker and ticker in normalized_fired:
            fatal.append(error)
        else:
            tolerated.append(error)
    return fatal, tolerated


def _fetch_error_summaries(
    fetch_errors: List[Dict[str, Any]],
    *,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    return [
        {
            "ticker": str(error.get("ticker") or "").strip().upper() or None,
            "stage": error.get("stage"),
        }
        for error in fetch_errors[:limit]
    ]


def _record_response_lineage(
    session: Session,
    resp: Any,
    *,
    job_run_id: str,
    raw_payload: Dict[str, Any],
) -> Any:
    return record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash,
        request_timestamp=resp.lineage.request_timestamp,
        freshness_seconds=resp.lineage.freshness_seconds,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )


def _base_metrics(
    session_resolution: Any,
    *,
    scan_id: Optional[str],
    included_universe_size: int,
    from_date: date,
    to_date: date,
    included_market_cap_bucket_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return {
        "decision_date": session_resolution.decision_date,
        "evidence_session_date": session_resolution.evidence_session_date,
        "next_execution_session": session_resolution.next_execution_session,
        "is_premarket_decision_window": session_resolution.is_premarket_decision_window,
        "session_resolution": asdict(session_resolution),
        "canonical_scan_id": scan_id,
        "included_universe_size": included_universe_size,
        "included_market_cap_bucket_counts": included_market_cap_bucket_counts or {},
        "form4_from_date": from_date.isoformat(),
        "form4_to_date": to_date.isoformat(),
    }


def _assembly_metrics(assembly: Any) -> Dict[str, Any]:
    return {
        "pattern_id": assembly.pattern_id,
        "assembled_count": assembly.assembled_count,
        "rejected_count": assembly.rejected_count,
        "insufficient_count": assembly.insufficient_count,
        "diagnostic_count": len(assembly.diagnostics),
        "diagnostics": [asdict(d) for d in assembly.diagnostics[:50]],
        "rejected_field_count": len(assembly.rejected_fields),
    }


def _accession_owner_ciks(evidence_rows: List[Any]) -> Dict[str, str]:
    owners_by_accession: Dict[str, set[str]] = {}
    for row in evidence_rows:
        if row.filing_accession_number and row.insider_cik:
            owners_by_accession.setdefault(row.filing_accession_number, set()).add(row.insider_cik)
    return {
        accession: next(iter(ciks))
        for accession, ciks in owners_by_accession.items()
        if len(ciks) == 1
    }


def _dedupe_key(evidence: Any) -> tuple:
    return (
        evidence.filing_accession_number,
        evidence.insider_cik,
        evidence.transaction_date,
        evidence.transaction_code,
        evidence.transaction_shares,
        evidence.transaction_price_per_share,
    )


def _provider_error(error: Any) -> Dict[str, Any]:
    if error is None:
        return {}
    return {
        "error_type": getattr(error, "error_type", None),
        "status_code": getattr(error, "status_code", None),
        "message": getattr(error, "message", None),
        "retryable": getattr(error, "retryable", None),
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _ensure_aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
