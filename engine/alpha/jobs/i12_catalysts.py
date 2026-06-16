"""PIT-safe catalyst tagging for the I12 historical corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse
from alpha.data.edgar import (
    _cik10,
    _parse_recent_filings,
)
from alpha.db.models import SecurityIdentitySnapshot
from alpha.jobs.watchdog import (
    ProviderOutageCircuitBreaker,
    WatchdogState,
    call_with_daemon_deadline,
)


CATALYST_STATUS_IMPLEMENTED = "implemented_pit_filtered"
TAG_DILUTION_OFFERING = "dilution_offering"
TAG_NT_LATE_FILER = "nt_late_filer"
TAG_FDA_CLINICAL = "fda_clinical"
TAG_COMPLIANCE_LISTING = "compliance_listing"
TAG_MNA_STRATEGIC = "mna_strategic"

CATALYST_FEATURE_DEFAULTS = {
    "catalyst_dilution_avoid": False,
    "catalyst_nt_late_filer": False,
    "catalyst_fda_amplifier": False,
    "catalyst_compliance_amplifier": False,
}

_DILUTION_FORMS = ("424B", "S-1", "S-3", "F-1", "F-3")
_NT_FORMS = ("NT 10-K", "NT 10-Q")
_FDA_RE = re.compile(r"\b(fda|clinical|trial|phase\s*[123]|pdufa|crl|approval)\b", re.I)
_COMPLIANCE_RE = re.compile(r"\b(nasdaq|nyse|listing|compliance|deficien|delist|minimum bid)\b", re.I)
_MNA_RE = re.compile(r"\b(merger|acquisition|strategic alternatives|buyout|takeover)\b", re.I)
_OFFERING_RE = re.compile(r"\b(atm|registered direct|offering|shelf|prospectus|warrant)\b", re.I)


@dataclass(frozen=True)
class CatalystEvent:
    tag: str
    source: str
    timestamp: datetime
    summary: str
    provider_id: str | None = None


@dataclass(frozen=True)
class CatalystResult:
    tags: tuple[str, ...]
    features: dict[str, bool]
    source_status: str
    cutoff_timestamp: datetime | None
    cik: str | None = None
    events: tuple[CatalystEvent, ...] = ()
    source_counts: dict[str, int] = field(default_factory=dict)
    source_errors: tuple[dict[str, Any], ...] = ()

    def as_feature_updates(self) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "catalyst_tags": list(self.tags),
            "catalyst_source_status": self.source_status,
            "catalyst_cutoff_timestamp": _iso_utc(self.cutoff_timestamp),
            "catalyst_cik": self.cik,
            "catalyst_source_counts": dict(self.source_counts),
            "catalyst_events": [
                {
                    "tag": event.tag,
                    "source": event.source,
                    "timestamp": _iso_utc(event.timestamp),
                    "summary": event.summary,
                    "provider_id": event.provider_id,
                }
                for event in self.events
            ],
            "catalyst_source_errors": list(self.source_errors),
        }
        updates.update(self.features)
        return updates


def empty_catalyst_result(
    *,
    cutoff_timestamp: datetime | None,
    cik: str | None = None,
    source_counts: Mapping[str, int] | None = None,
    source_errors: Sequence[Mapping[str, Any]] | None = None,
) -> CatalystResult:
    return CatalystResult(
        tags=(),
        features=dict(CATALYST_FEATURE_DEFAULTS),
        source_status=CATALYST_STATUS_IMPLEMENTED,
        cutoff_timestamp=_aware_utc_or_none(cutoff_timestamp),
        cik=cik,
        events=(),
        source_counts=dict(source_counts or {}),
        source_errors=tuple(dict(error) for error in (source_errors or ())),
    )


def apply_i12_catalyst_result_to_feature_payload(
    payload: dict[str, Any],
    result: CatalystResult,
) -> dict[str, Any]:
    updated = dict(payload)
    updated.update(result.as_feature_updates())
    return updated


class I12CatalystResolver:
    """Resolve PIT-safe catalyst tags without per-document EDGAR crawls."""

    def __init__(
        self,
        *,
        session: Session,
        edgar_adapter: Any | None = None,
        polygon_news_adapter: Any | None = None,
        benzinga_news_adapter: Any | None = None,
        edgar_cache_dir: str | Path | None = None,
        lookback_days: int = 5,
        fetch_deadline_seconds: float | None = None,
        watchdog_state: WatchdogState | None = None,
    ) -> None:
        if lookback_days < 0:
            raise ValueError("lookback_days must be >= 0")
        if fetch_deadline_seconds is not None and fetch_deadline_seconds <= 0:
            raise ValueError("fetch_deadline_seconds must be > 0")
        self._session = session
        self._edgar = edgar_adapter
        self._polygon_news = polygon_news_adapter
        self._benzinga_news = benzinga_news_adapter
        self._edgar_cache_dir = Path(edgar_cache_dir) if edgar_cache_dir else None
        self._lookback_days = int(lookback_days)
        self._fetch_deadline_seconds = fetch_deadline_seconds
        self._watchdog_state = watchdog_state

    def resolve(
        self,
        *,
        ticker: str,
        cutoff_timestamp: datetime | None,
        trading_date: date,
    ) -> CatalystResult:
        cutoff = _aware_utc_or_none(cutoff_timestamp)
        if cutoff is None:
            return empty_catalyst_result(cutoff_timestamp=None)
        ticker = ticker.upper()
        cik = self._resolve_cik(ticker)
        window_start = datetime.combine(
            trading_date - timedelta(days=self._lookback_days),
            time.min,
            tzinfo=timezone.utc,
        )
        events: list[CatalystEvent] = []
        errors: list[dict[str, Any]] = []
        counts: dict[str, int] = {
            "edgar_filings_considered": 0,
            "edgar_filings_included": 0,
            "news_articles_considered": 0,
            "news_articles_included": 0,
        }
        if cik and self._edgar is not None:
            edgar_events, edgar_counts, edgar_errors = self._edgar_events(
                cik=cik,
                cutoff=cutoff,
                window_start=window_start,
            )
            events.extend(edgar_events)
            _merge_counts(counts, edgar_counts)
            errors.extend(edgar_errors)
        if self._polygon_news is not None:
            news_events, news_counts, news_errors = self._polygon_news_events(
                ticker=ticker,
                cutoff=cutoff,
                window_start=window_start,
            )
            events.extend(news_events)
            _merge_counts(counts, news_counts)
            errors.extend(news_errors)
        if self._benzinga_news is not None:
            news_events, news_counts, news_errors = self._benzinga_news_events(
                ticker=ticker,
                cutoff=cutoff,
                trading_date=trading_date,
                window_start=window_start,
            )
            events.extend(news_events)
            _merge_counts(counts, news_counts)
            errors.extend(news_errors)
        return _result_from_events(
            events,
            cutoff_timestamp=cutoff,
            cik=cik,
            source_counts=counts,
            source_errors=errors,
        )

    def _resolve_cik(self, ticker: str) -> str | None:
        row = (
            self._session.query(SecurityIdentitySnapshot)
            .filter(SecurityIdentitySnapshot.ticker == ticker.upper())
            .order_by(
                SecurityIdentitySnapshot.active.desc(),
                SecurityIdentitySnapshot.asof_timestamp.desc().nullslast(),
                SecurityIdentitySnapshot.security_identity_snapshot_id.desc(),
            )
            .first()
        )
        if row is None:
            return None
        return _cik10(row.cik)

    def _edgar_events(
        self,
        *,
        cik: str,
        cutoff: datetime,
        window_start: datetime,
    ) -> tuple[list[CatalystEvent], dict[str, int], list[dict[str, Any]]]:
        payload, error = self._load_edgar_submissions(cik, cutoff=cutoff)
        if error is not None:
            return [], {}, [error]
        filings = _parse_recent_filings(cik, payload or {})
        events: list[CatalystEvent] = []
        considered = 0
        for filing in filings:
            accepted = _aware_utc_or_none(filing.acceptance_datetime)
            if accepted is None or not (window_start <= accepted < cutoff):
                continue
            considered += 1
            for tag in _tags_for_filing(filing.form, filing.primary_doc_description):
                events.append(
                    CatalystEvent(
                        tag=tag,
                        source="sec_edgar_submissions",
                        timestamp=accepted,
                        summary=f"{filing.form} {filing.accession_number}",
                        provider_id=filing.accession_number,
                    )
                )
        return events, {
            "edgar_filings_considered": considered,
            "edgar_filings_included": len(events),
        }, []

    def _load_edgar_submissions(
        self,
        cik: str,
        *,
        cutoff: datetime,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        cached = self._read_cached_submissions(cik)
        if cached is not None:
            return cached, None
        if self._edgar is None:
            return None, {"source": "sec_edgar_submissions", "error": "adapter_unavailable"}

        def _fetch() -> AdapterResponse[Any]:
            return self._edgar.get_company_submissions(cik, asof=cutoff)

        try:
            resp = self._call_provider(
                _fetch,
                thread_name="i12-catalyst-edgar-submissions",
                context={"stage": "edgar_submissions", "cik": cik},
            )
        except ProviderOutageCircuitBreaker:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed for catalyst enrichment
            return None, {
                "source": "sec_edgar_submissions",
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
        if not resp.ok:
            return None, {
                "source": "sec_edgar_submissions",
                "error": getattr(resp.error, "error_type", "provider_error"),
                "message": getattr(resp.error, "message", None),
            }
        data = dict(resp.data or {})
        self._write_cached_submissions(cik, data)
        return data, None

    def _read_cached_submissions(self, cik: str) -> dict[str, Any] | None:
        path = self._cache_path(cik)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cached_submissions(self, cik: str, payload: Mapping[str, Any]) -> None:
        path = self._cache_path(cik)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, default=str))
        tmp.replace(path)

    def _cache_path(self, cik: str) -> Path | None:
        if self._edgar_cache_dir is None:
            return None
        cik10 = _cik10(cik)
        if cik10 is None:
            return None
        return self._edgar_cache_dir / f"CIK{cik10}.json"

    def _polygon_news_events(
        self,
        *,
        ticker: str,
        cutoff: datetime,
        window_start: datetime,
    ) -> tuple[list[CatalystEvent], dict[str, int], list[dict[str, Any]]]:
        def _fetch() -> AdapterResponse[Any]:
            return self._polygon_news.get_news(
                ticker=ticker,
                published_utc_from=_iso_utc(window_start),
                published_utc_to=_iso_utc(cutoff),
                limit=100,
                max_pages=1,
                asof=cutoff,
            )

        try:
            resp = self._call_provider(
                _fetch,
                thread_name="i12-catalyst-polygon-news",
                context={"stage": "polygon_news", "ticker": ticker},
            )
        except ProviderOutageCircuitBreaker:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed for optional news source
            return [], {}, [{"source": "polygon_news", "error": exc.__class__.__name__, "message": str(exc)}]
        if not resp.ok:
            return [], {}, [{
                "source": "polygon_news",
                "error": getattr(resp.error, "error_type", "provider_error"),
                "message": getattr(resp.error, "message", None),
            }]
        return _news_events_from_articles(
            resp.data or (),
            source="polygon_news",
            cutoff=cutoff,
            published_attr="published_utc",
        )

    def _benzinga_news_events(
        self,
        *,
        ticker: str,
        cutoff: datetime,
        trading_date: date,
        window_start: datetime,
    ) -> tuple[list[CatalystEvent], dict[str, int], list[dict[str, Any]]]:
        def _fetch() -> AdapterResponse[Any]:
            return self._benzinga_news.get_news(
                tickers=ticker,
                date_from=window_start.date().isoformat(),
                date_to=trading_date.isoformat(),
                limit=100,
                asof=cutoff,
            )

        try:
            resp = self._call_provider(
                _fetch,
                thread_name="i12-catalyst-benzinga-news",
                context={"stage": "benzinga_news", "ticker": ticker},
            )
        except ProviderOutageCircuitBreaker:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed for optional news source
            return [], {}, [{"source": "benzinga_news", "error": exc.__class__.__name__, "message": str(exc)}]
        if not resp.ok:
            return [], {}, [{
                "source": "benzinga_news",
                "error": getattr(resp.error, "error_type", "provider_error"),
                "message": getattr(resp.error, "message", None),
            }]
        return _news_events_from_articles(
            resp.data or (),
            source="benzinga_news",
            cutoff=cutoff,
            published_attr="published",
        )

    def _call_provider(
        self,
        func: Any,
        *,
        thread_name: str,
        context: dict[str, Any],
    ) -> Any:
        if self._fetch_deadline_seconds is None:
            return func()
        return call_with_daemon_deadline(
            func,
            timeout_seconds=self._fetch_deadline_seconds,
            thread_name=thread_name,
            state=self._watchdog_state,
            context=context,
        )


def _result_from_events(
    events: Sequence[CatalystEvent],
    *,
    cutoff_timestamp: datetime,
    cik: str | None,
    source_counts: Mapping[str, int],
    source_errors: Sequence[Mapping[str, Any]],
) -> CatalystResult:
    ordered = tuple(sorted(events, key=lambda event: (event.timestamp, event.tag, event.source)))
    tags = tuple(sorted({event.tag for event in ordered}))
    features = {
        "catalyst_dilution_avoid": TAG_DILUTION_OFFERING in tags,
        "catalyst_nt_late_filer": TAG_NT_LATE_FILER in tags,
        "catalyst_fda_amplifier": TAG_FDA_CLINICAL in tags,
        "catalyst_compliance_amplifier": TAG_COMPLIANCE_LISTING in tags,
    }
    return CatalystResult(
        tags=tags,
        features=features,
        source_status=CATALYST_STATUS_IMPLEMENTED,
        cutoff_timestamp=cutoff_timestamp,
        cik=cik,
        events=ordered,
        source_counts=dict(source_counts),
        source_errors=tuple(dict(error) for error in source_errors),
    )


def _tags_for_filing(form: str | None, description: str | None) -> tuple[str, ...]:
    normalized = str(form or "").strip().upper()
    text = f"{normalized} {description or ''}"
    tags: list[str] = []
    if normalized.startswith(_DILUTION_FORMS) or _OFFERING_RE.search(text):
        tags.append(TAG_DILUTION_OFFERING)
    if normalized.startswith(_NT_FORMS):
        tags.append(TAG_NT_LATE_FILER)
    return tuple(tags)


def _news_events_from_articles(
    articles: Sequence[Any],
    *,
    source: str,
    cutoff: datetime,
    published_attr: str,
) -> tuple[list[CatalystEvent], dict[str, int], list[dict[str, Any]]]:
    events: list[CatalystEvent] = []
    considered = 0
    for article in articles:
        published = _article_published_at(article, published_attr)
        if published is None or not published < cutoff:
            continue
        considered += 1
        text = _article_text(article)
        tags = _tags_for_news_text(text)
        for tag in tags:
            events.append(
                CatalystEvent(
                    tag=tag,
                    source=source,
                    timestamp=published,
                    summary=_article_title(article)[:200],
                    provider_id=_article_id(article),
                )
            )
    return events, {
        "news_articles_considered": considered,
        "news_articles_included": len(events),
    }, []


def _tags_for_news_text(text: str) -> tuple[str, ...]:
    tags: list[str] = []
    if _FDA_RE.search(text):
        tags.append(TAG_FDA_CLINICAL)
    if _COMPLIANCE_RE.search(text):
        tags.append(TAG_COMPLIANCE_LISTING)
    if _MNA_RE.search(text):
        tags.append(TAG_MNA_STRATEGIC)
    return tuple(tags)


def _article_text(article: Any) -> str:
    parts: list[str] = []
    for attr in ("title", "description", "body", "teaser"):
        value = getattr(article, attr, None)
        if value:
            parts.append(str(value))
    for attr in ("keywords", "tags", "categories"):
        value = getattr(article, attr, None)
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value if item)
    raw = getattr(article, "raw", None)
    if isinstance(raw, dict):
        parts.extend(str(value) for key, value in raw.items() if key in {"title", "description", "summary"})
    return " ".join(parts)


def _article_title(article: Any) -> str:
    return str(getattr(article, "title", None) or getattr(article, "description", None) or "")


def _article_id(article: Any) -> str | None:
    value = getattr(article, "id", None)
    return str(value) if value is not None else None


def _article_published_at(article: Any, attr: str) -> datetime | None:
    value = getattr(article, attr, None)
    if isinstance(value, datetime):
        return _aware_utc_or_none(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _aware_utc_or_none(parsed)
    return None


def _aware_utc_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _aware_utc_or_none(value).isoformat().replace("+00:00", "Z")


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)
