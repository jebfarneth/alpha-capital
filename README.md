# Alpha Capital

Production-grade alpha research and signal-accumulation infrastructure for U.S. equities.

Alpha Capital is a backend-first systematic trading engine. It is built around one idea: a trading system is only useful if it can prove what it knew at decision time, preserve the full evidence trail, measure what happened afterward without survivorship bias, and feed those outcomes back into an optimizer or ML layer.

This repository is not a charting app and it is not a notebook backtest. It is the operating spine of a quant platform:

```text
Universe -> Feature Assembly -> Pattern Detection -> Signal Registry
         -> Market Path Features -> Forward Measurement -> Scoreboard
         -> Candidate Selection -> Optimizer -> Execution
```

The current system runs live signal accumulation, persists evidence to Postgres, tracks provider lineage, and is being expanded into a historical replay + ML training corpus. Broker execution is intentionally downstream of measurement. The engine is collecting the truth set first.

---

## Current Snapshot

As of June 2026, Alpha Capital has moved beyond a single-pattern prototype into a multi-pattern measurement platform.

| Area | Current state |
|---|---|
| Runtime | Python 3.9+ engine with SQLAlchemy, Alembic, pytest, Postgres/Supabase, and provider adapters. |
| Production DB | Supabase/Postgres canonical database with Alembic-managed schema. |
| Production compute | Google Compute Engine VM scheduled after the market close, monitored by Healthchecks.io. |
| Universe | Live FMP/Polygon-built operating universe widened from microcap-only to `$30M-$5B`; latest proven wide build included 2,621 names. |
| Live accumulation | M4 is the original live lane; M1 and M2 producer paths are built and integrated behind explicit rollout gates. |
| M2 | SEC Form 4 backfill/warm-start path built; widened-universe SEC seed is intentionally staged before the expanded universe is enabled with M2. |
| M3 | Sector-rotation detector and PIT sector-history pipeline built, default-off until coverage and governance gates are met. |
| Market-path features | Durable per-signal/per-day feature store built for base EOD, rich EOD technicals, relative features, ML context fields, lineage, hashes, and null/status auditability. |
| Testing | Current full-suite baseline is over 2,000 passing Python tests with scratch-schema and real-provider integration probes. |
| Safety posture | Public writes are guarded; scratch schemas are isolated; M3 remains default-off; market-path failures are observability-only; M4/M1/M2 producer behavior is protected by regression tests. |

The project is currently in signal accumulation, data-quality hardening, historical replay, and ML feature-corpus construction. It is not yet live broker execution.

---

## What Makes It Different

Most trading projects fail quietly. The backtest looks good because the data is not point-in-time, delisted names disappeared, a provider error became a clean empty result, or the model trained on a label that could not have been known in production.

Alpha Capital is engineered against those failure modes.

| Failure mode | Engine response |
|---|---|
| Lookahead | Assemblers use explicit decision/evidence/execution clocks and PIT cutoffs. |
| Survivorship bias | Delisting/listing authority is modeled separately from price providers. |
| Provider partial failure | Source attempts are recorded; provider errors are not silently collapsed into no-data. |
| Duplicate signals | Signal identity hashes and uniqueness constraints protect the registry. |
| Stale schema | Canonical runs verify Alembic state before writing. |
| Scratch accidents | Scratch mode refuses public, creates isolated schemas, and can be dropped/verified. |
| ML feature contamination | Feature rows carry lineage, hashes, status flags, pattern identity, and missing-source markers. |
| Default-off features leaking into prod | M3 and other prospective work are gated by flags, migration order, and tests. |

The engineering standard is not "the detector fired." The standard is "we can audit every input that caused it to fire and every label later used to score it."

---

## System Architecture

### Engine

The core engine lives under `engine/alpha`.

| Module | Purpose |
|---|---|
| `alpha.data` | Typed adapters for FMP, Polygon, Benzinga, SEC EDGAR, Nasdaq, and Alpaca. |
| `alpha.db` | SQLAlchemy models and Postgres session management. |
| `alpha.evidence` | Evidence job/run helpers, lineage, and schema-correct persistence. |
| `alpha.patterns` | Pure pattern detectors and shared guard logic. |
| `alpha.assembly` | Pattern-specific feature assembly and market-data payload builders. |
| `alpha.jobs` | Production/scratch entrypoints, orchestration, forward measurement, market-path features, universe builds, and scoreboard jobs. |
| `engine/migrations` | Alembic schema history. |
| `engine/tests` | Unit, integration, schema, provider, and orchestration regression tests. |

### Production Flow

```text
1. Resolve market calendar and decision/evidence/execution sessions.
2. Build or reuse the canonical operating universe.
3. Run enabled pattern producers.
4. Persist fired signals and no-signal evidence.
5. Persist provider attempts, feature snapshots, data lineage, and run metrics.
6. Enrich each signal with market-path features across its forward window.
7. Measure forward returns when the observation window matures.
8. Feed labels and feature rows into scoreboard / ML research.
```

The canonical runner owns market-calendar logic. Cron can fire blindly; the runner exits cleanly on non-trading days and refuses unsafe write targets.

---

## Data Model

The schema is designed for replayability and auditability rather than just storing "signals."

| Data family | Examples |
|---|---|
| Evidence jobs | `evidence_jobs`, `evidence_job_runs`, structured metrics, run statuses. |
| Lineage | `data_lineage` rows carrying provider, endpoint, payload hashes, quality flags, and source attempts. |
| Universe | `universe_scans`, `canonical_universe_scans`, `universe_snapshots`, `security_profiles`, identity snapshots. |
| Signals | `signal_registry`, detector output, feature snapshots, signal identity hashes, pattern IDs. |
| Forward measurement | Forward-return observations, forward path rows, maturity states, provider-revision review states. |
| Market-path ML features | One row per signal per market day with base EOD, rich EOD technical, relative, context, status, hash, and lineage fields. |
| M2 insider data | SEC Form 4 transaction history, source-gated M2/M2U signal assembly, SEC fetch-coverage ledger. |
| M3 sector data | PIT sector-assignment history, sector returns, sector-change logs, validation metadata, default-off until coverage gates. |

The platform treats nulls as first-class evidence. A null with `missing_intraday_bars` is different from a null with `provider_error`, and both are different from a value that was computed.

---

## Market-Path Feature Store

`market_path_features` is the ML data spine. It stores one row per signal per market session and is intentionally pattern-agnostic while preserving pattern identity. The current public proof covers M4 and M1 signal-day plus forward-path rows for 2026-06-01 through 2026-06-05; the next planned extension is pre-signal setup context so models can learn what strong winners looked like before the signal fired.

The intended ML corpus has three separate zones:

| Zone | Purpose | Leakage rule |
|---|---|---|
| Pre-signal setup | T-60 through T-1 context before a signal fires. | Predictors only; no signal-day or forward-path values. |
| Signal-day anatomy | T0 close/candle/volume/context and day-zero behavior. | Usable only when the decision clock could have known it. |
| Forward outcome path | T+1 through horizon returns, MFE/MAE, barrier/exit labels. | Labels and diagnostics only; never same-row predictors. |

Already built locally:

| Feature group | Examples |
|---|---|
| Base EOD | OHLCV, dollar volume, 20d/60d medians, volume expansion, gap, open-to-close, high/low-from-open, entry-relative returns. |
| Risk/path | Sigma, effective hard-stop proxy, liquidity proxy, feature role, path sequence. |
| Breakout anatomy | Prior 52-week high, breakout extension, open/high/close vs. high, gap-over-breakout, close-held-above-breakout. |
| Candle/range | Close-location value, upper/lower wick ratios, true range, ATR, range expansion. |
| Volume | Volume and dollar-volume z-scores, acceleration, expansion ranks and percentiles. |
| Compression/base | Realized volatility, base range, base drawdown over multiple windows. |
| Trend/momentum | SMA distances, 5d/20d/60d momentum, prior high touches, failed-breakout counts. |
| VWAP | FMP raw VWAP and OHLC vs. VWAP fields when available. |
| Relative context | SPY/QQQ/IWM benchmark returns, benchmark relative strength, IWM-vs-SPY and QQQ-vs-SPY regime fields. |
| Sector relative | Sector ETF mapping and relative strength when PIT sector history exists. |
| Market regime | SPY/IWM realized-volatility proxy, status fields for missing breadth/VIX sources. |
| Intraday placeholders | Opening range, first-window returns, intraday VWAP, time-bucket volume, MFE/MAE timestamps, T1-before-stop ordering, all null/status until real intraday bars are wired. |
| Liquidity/execution placeholders | Quote spread, quote age, depth, executable-entry proxy, participation, halt/offering risk, all source-stamped. |
| Supply/squeeze placeholders | Float, shares, turnover, short volume, short interest, days-to-cover, borrow fee, with PIT/source status. |
| Catalyst/cross-pattern | M2 insider overlap, co-fire flags, overlap counts, strongest-overlap pattern, and source-stamped catalyst placeholders. |
| Classic technicals | RSI, ADX/DMI, Bollinger, Keltner, MACD histogram, OBV, accumulation/distribution, Chaikin money flow, stochastic oscillator. |

Feature rows carry output hashes, input/window metadata, lineage IDs, and explicit status fields so the ML layer can distinguish "not available yet" from "not applicable" from "provider failed."

---

## Pattern Portfolio

Alpha Capital uses a pattern roster rather than a single monolithic signal. Each detector is separated from the assembler that feeds it, which makes it possible to audit detector logic independently from data availability.

| Pattern | Thesis | Current status |
|---|---|---|
| M1 - Post-Earnings Drift | Earnings surprise continuation and drift. | Producer/assembler/detector built; integrated into measurement spine; early evidence suggests signal-day/intraday timing matters more than next-session entry. |
| M2 - Insider Cluster | Opportunistic insider buying clusters from SEC Form 4s. | Producer/assembler/detector built; source-gating fixed; paused in production until the widened-universe SEC seed is made resumable/batched. |
| M2U - Insider Cluster Upside | M2 variant focused on upside cluster behavior. | Built with M2 path. |
| M3 - Sector Rotation | PIT sector-relative return and cross-sectional sector rank. | Detector and sector-history pipeline built; default-off pending coverage gates. |
| M4 - 52-Week High | EOD close breaks prior 52-week high. | Original live signal-accumulation lane. |
| M5 - Failed Breakdown Reversal | Failed downside break with reversal dynamics. | Detector built. |
| M6 - Volatility Compression Breakout | Low-volatility compression followed by expansion. | Detector built. |
| M7 - Pure Technical ML | ML-recognized multi-day technical setups under Reality-Check governance. | Detector scaffold built. |
| I1 - Gap and Go | Gap continuation with intraday confirmation. | Detector built; intraday data remains a blocker for production. |
| I8 - Opening Range Breakout | Opening range breakout behavior. | Detector built; intraday data remains a blocker for production. |
| I11 - 52-Week High Breakout | Intraday twin of M4: cross prior high live, volume-confirmed, day-0 capture. | Spec/prospective; strong daily-proxy research, but blocked on true intraday bars and execution ordering proof. |

Additional event and intraday overlays are specified in the vault but intentionally not promoted until their data sources, PIT proof, and execution assumptions are validated.

---

## M4 And I11

M4 and I11 are intentionally related but not the same.

| Pattern | Clock | Entry idea | Main risk |
|---|---|---|---|
| M4 | End-of-day | Next session after a confirmed close above the prior 52-week high. | Gives up day-0 surge. |
| I11 | Intraday | Enter when price crosses the prior 52-week high live, volume-confirmed. | Requires real intraday ordering, spreads, and executable fills. |

M4 is the stable accumulation lane. I11 is the prospective edge candidate that may capture the move M4 structurally misses. The current research view is that I11 looks promising, but daily OHLC proxy is not enough; the deciding proof must come from real intraday bars and executable spread/fill assumptions.

---

## M2 Insider Pipeline

M2 is built around SEC Form 4 transactions.

Key design decisions:

- Transaction IDs are deterministic hashes of SEC content.
- Nightly upserts by transaction ID make warm-start seeding safe.
- Source-gating prevents transient fetch errors on non-firing tickers from false-failing the canonical nightly.
- Fetch errors on tickers that actually fire are fatal because the signal would be certified on incomplete evidence.
- A SEC fetch-coverage ledger warms tickers even when they have no Form 4 history, preventing repeated full-history cold pulls.

The widened `$30M-$5B` universe materially increases the number of tickers M2 must scan. The first long SEC warm seed exposed a runtime/DB reliability problem, so production currently skips M2 while the seed path is redesigned as a resumable/batched job.

---

## M3 Sector Rotation

M3 is the point-in-time sector-rotation lane.

It includes:

- Polygon SIC-as-of sector-history pipeline.
- Versioned SIC-to-sector mapping.
- Exact-as-of sector intervals with overlap protection.
- Value-weighted formation-cohort sector returns.
- PIT proof flags consumed by the detector.
- Shadow-only behavior until coverage gates are satisfied.
- M3S shadow pattern handling for undercovered sectors.

M3 remains default-off. The schema and code are built, but production firing is blocked until the PIT history and coverage requirements are satisfied.

---

## Provider Authority

Alpha Capital separates data convenience from data authority.

| Provider | Current role |
|---|---|
| FMP | Universe screening, profiles, daily OHLCV, EOD replay/backfill, fundamentals where appropriate. |
| Polygon | Identity, corporate actions, news, short volume/short interest/float availability, daily sanity checks, PIT ticker details. |
| Benzinga | Catalyst/news calendars, WIIMs, earnings, ratings, offerings, insider/news context. |
| SEC EDGAR | Official filing authority for Form 4s, Form 25/25-NSE, acceptance-time PIT handling. |
| Nasdaq | Listing/halt/archive authority pipeline. |
| Alpaca | Broker/execution adapter scaffold; production execution is intentionally not live yet. |

Provider errors are observable states. The engine records attempts and distinguishes `no_data`, `provider_error`, `parse_error`, `validation_error`, and PIT exclusion states.

---

## Point-In-Time Discipline

The system uses multiple clocks because trading evidence lives on multiple clocks.

| Clock | Meaning |
|---|---|
| Decision date | Session for which the engine is deciding signal eligibility. |
| Evidence session | Market session whose close is allowed into EOD features. |
| PIT cutoff | Timestamp after which provider filings/news cannot be used. |
| Execution session | Earliest realistic next-session entry point. |
| Forward window | Post-signal sessions used for measurement labels. |

Examples of PIT handling:

- SEC EDGAR acceptance timestamps are normalized to the correct knowledge time before comparison to cutoff.
- M3 sector returns use formation-date sector membership, not current sector membership.
- Market-path trailing features use prior bars only unless explicitly defined as same-day post-close fields.
- Relative ranks are isolated by feature date and pattern ID, preventing accidental cross-pattern pooling.
- Missing PIT sector history produces `null + status`, not a guessed sector.

---

## Machine Learning Strategy

The ML objective is not "ask AI what to buy." The objective is to build a clean, labeled, point-in-time feature corpus that can support supervised ranking, calibration, and allocation.

The design principles:

1. Preserve `pattern_id` on every feature and label row.
2. Train per-pattern models first.
3. Allow cross-pattern models only when `pattern_id` is an explicit feature and performance is reported per pattern.
4. Never pool M1/M2/M3/M4/I11 blindly.
5. Treat source-missing flags as model inputs, not data-cleaning noise.
6. Backfill historical cohorts only with survivorship-correct PIT universe reconstruction.
7. Prefer robust ranking and capital allocation over a single binary classifier.

The immediate research loop is:

```text
Build historical PIT universe by date
-> replay pattern fires by date
-> persist pre-signal setup context
-> persist signal-day anatomy
-> reconstruct forward paths and labels
-> train ranking/exit/selection models
-> test by date, universe bucket, liquidity bucket, and pattern
```

This is also why the current system is collecting broad feature rows before broker execution. The M-patterns are useful as event and evidence generators even if their naive next-session trade rules are not profitable. The likely trading edge is expected to come from point-in-time meta-labeling and intraday derivative patterns such as I11, after the corpus proves which day-zero setups actually follow through.

---

## Historical Replay Plan

The historical replay path is designed to answer: "What would the live engine have fired on each past date?"

Target path:

1. Load a point-in-time delisted-company source before public replay.
2. Reconstruct the point-in-time universe for each historical session.
3. Include names alive on that date, including names later delisted.
4. Exclude names not yet IPO'd, already delisted, foreign, or non-common securities.
5. Pull historical OHLCV and source-backed context.
6. Recompute the exact pattern gates from the code, not from an approximation.
7. Persist reconstructed signals with provenance.
8. Compute forward labels immediately because the 15-session windows are already in the past.
9. Keep reconstructed rows separate from live-collected rows.

For bounded replay windows, the FMP delisted-company ingest supports a date cutoff such as `--stop-after-delisted-before 2026-01-01`. The cohort runner accepts that bounded source only when the recorded cutoff covers the requested replay start date; otherwise public replay fails closed.

The immediate operational target is Jan-May 2026 M4 cohort reconstruction, then post-signal market-path enrichment, then pre-signal context rows for the same reconstructed signals.

---

## Operations And Safety

Production writes are intentionally hard to do accidentally.

| Guard | Purpose |
|---|---|
| `--live` + confirmation | Canonical writes require explicit live mode and confirmation. |
| PostgreSQL-only canonical | SQLite is refused for canonical writes. |
| No `ALPHA_DB_SCHEMA` in canonical | Prevents accidentally writing production through a scratch setting. |
| Scratch schema required for scratch | Scratch runs must provide a schema and refuse `public`. |
| Alembic head/runtime checks | Production writes require the expected schema state. |
| Non-trading day no-op | Weekends/holidays exit cleanly before provider/DB work. |
| Source-gated health | Provider errors fail only when they affect fired-source evidence. |
| Health report | Canonical jobs emit structured health and metrics. |
| Idempotency checks | Reruns update existing rows rather than duplicating signal/path rows. |

Market-path feature collection is deliberately observability-only: a feature enrichment issue should be visible, but it should not falsely mark M4/M1/M2 signal production as failed after those producers already persisted correctly.

---

## Testing And Audit Culture

The test suite is large because the system is data-critical.

Current themes:

- schema completeness tests
- Alembic head and migration-chain tests
- canonical write-target guards
- scratch-schema integration tests
- provider parser tests
- market-calendar tests
- detector and assembler tests
- no-lookahead tests
- idempotency tests
- source-gating tests
- real Postgres scratch probes
- real provider probes when fixture-only tests are insufficient

The workflow uses dual adversarial audits. One terminal may pass a change while another finds a blocker; the stricter verdict wins. Findings are labeled `NEEDS FIXING`, fixed in a new pass, then re-audited.

Recent examples of issues caught by this process:

- M2 fetch errors on non-firing tickers falsely failing the canonical nightly.
- M3 default-off failures leaking into canonical exit code.
- M3 shadow pattern rows persisting outside downstream contracts.
- Sector-history interval overlap on out-of-order writes.
- Market-path feature lineage containing batch bars that needed row-scope clarification.
- Alembic migration order accidentally forcing M3 schema before default-off public rollout.
- Missing 1d benchmark fields in the relative-feature surface.

The point is not theater. In a trading system, silent optimism is the expensive failure.

---

## Repository Layout

```text
alpha-capital/
  engine/
    alpha/
      assembly/        Pattern feature assembly
      data/            Provider adapters
      db/              SQLAlchemy models and engine
      evidence/        Job/run and lineage helpers
      jobs/            Production and scratch jobs
      patterns/        Pure detectors and shared guards
    migrations/        Alembic schema history
    tests/             Python test suite
  client/              Dashboard frontend scaffold
  server/              API/server scaffold
  docs/                Dashboard mockups and support docs
  expansion.md         Market-path feature expansion notes
  m4_backfill_task_list.md
```

---

## Running Locally

The engine expects provider credentials and a database URL in `engine/.env`.

```bash
cd engine
uv run pytest -q
uv run alembic heads
```

Common entrypoints:

```bash
# Build / inspect the universe
uv run python -m alpha.jobs.run_universe --live --trading-date YYYY-MM-DD

# Run M4 daily path
uv run python -m alpha.jobs.run_m4_daily --live --run-timestamp YYYY-MM-DDTHH:MM:SS-04:00

# Run the canonical orchestrator in dry-run mode
uv run python -m alpha.jobs.run_nightly_canonical --dry-run

# Run market-path feature enrichment
uv run python -m alpha.jobs.run_market_path_features --schema scratch_schema
```

Canonical production writes require stricter flags and should only be run against the intended Supabase/Postgres target.

---

## Engineering Skills Demonstrated

This project exercises the kind of backend and data-engineering work that is hard to fake in toy apps:

- production Python service design
- SQLAlchemy/Postgres schema design
- Alembic migration governance
- point-in-time data modeling
- provider adapter hardening
- financial-data lineage and replay
- idempotent job orchestration
- canonical vs. scratch environment safety
- real-provider integration testing
- ML feature-store design
- audit-heavy release process
- cloud scheduling and monitoring
- systematic-trading risk discipline

The system is intentionally built as infrastructure first. The edge can only be trusted if the data path, labels, and operational controls are trustworthy.
