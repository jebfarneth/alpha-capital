# Alpha Capital

Alpha Capital is an autonomous systematic-trading engine for US small-cap equities. It is built backend-first around point-in-time data ingestion, auditable feature assembly, pattern detection, evidence capture, forward-return measurement, and eventually execution.

This is not a dashboard demo or a notebook backtest. It is a production-oriented Python/Postgres system designed to answer one hard question: can a small, disciplined trading engine generate and measure real signals without leaking future information, losing lineage, or hiding bad outcomes?

## Current Milestone

As of May 30, 2026, the first live accumulation lane is running:

- **Canonical M4 accumulation is live and unattended.** A cloud VM runs the canonical M4 daily runner on a weekday evening cron after provider finalization, with non-trading-day no-op behavior and dead-man monitoring.
- **M4 52-week-high signals are accumulating with frozen context.** The live universe is roughly 671 US-listed small-cap common stocks, and the first live cohort entered on May 28, 2026.
- **Forward-return infrastructure is committed but intentionally gated.** The system persists forward-return observations and 15-session daily path rows, but official forward scoring waits for EDGAR/Nasdaq survivorship authority.
- **SEC EDGAR adapter plumbing is now built.** The adapter supports company CIK/ticker mapping, submissions retrieval, PIT-filtered filing rows, and Form 25 / 25-NSE survivorship-event plumbing.
- **Test baseline:** `1644` Python tests passing, single Alembic head `b4c5d6e7f901`.

The next engineering frontier is wiring EDGAR into survivorship classification and building Nasdaq halt/listing authority so the first M4 cohort can be scored when it matures.

## Why This Project Is Technically Interesting

Alpha Capital is a compact version of the infrastructure problems that show up in real quant and trading systems:

- **Point-in-time correctness:** every adapter call, feature, signal, and report has explicit timestamps and lineage so the system can distinguish what was known at decision time from what became known later.
- **Evidence-first design:** signals, no-signal evidence, data lineage, provider attempts, job runs, forward paths, and validation artifacts are persisted instead of inferred from logs.
- **Provider authority separation:** FMP, Polygon, Benzinga, SEC EDGAR, Nasdaq, and Alpaca are not treated as interchangeable. Each provider has a defined authority level and failure mode.
- **Adversarial acceptance standard:** features are not accepted because unit tests pass. They go through fixture tests, live-read probes, scratch-schema rehearsals, secret-scan checks, PIT audits, and failure-direction audits.
- **Small-account realism:** the system is designed for a cash account starting around $1K, so liquidity, slippage, settlement, missed fills, and route class matter early.

## Architecture

The target system is a full quant operating loop:

```text
Universe -> Feature Assembly -> Pattern Detection -> Signal Registry
        -> Forward Measurement -> Candidate Builder -> Optimizer
        -> Execution -> Exit Manager -> Validation
```

Implemented production-path pieces include:

| Layer | Current state |
|---|---|
| Evidence spine | SQLAlchemy/Postgres models, Alembic migrations, job runs, lineage records, universe scans, feature snapshots, signal registry, forward-return observations, and forward path rows. |
| Universe builder | FMP screener/profile universe construction with security-type controls, market-cap bounds, price floor, exclusion telemetry, and Polygon identity enrichment. |
| Data adapters | Typed adapters for FMP, Polygon, Benzinga, Alpaca, and SEC EDGAR, all returning `AdapterResponse` with lineage/error metadata. |
| Detector framework | Callable detector registry and shared orchestration path for feature snapshots, no-signal evidence, signals, hashes, lineage IDs, and dedupe. |
| M4 daily lane | Production M4 52-week-high feature assembly, signal-context enrichment, canonical runner, rerun/idempotency checks, and health reporting. |
| Forward measurement | `price_fn` infrastructure, price-finality/drift handling, provider-revision sweep, 15-session path persistence, and sanctioned current-path reader. |
| Operations | GCP VM cron, Supabase/Postgres canonical DB, Healthchecks dead-man monitoring, guarded live/scratch CLI modes, and full-suite regression workflow. |

Downstream layers still to build include Nasdaq survivorship/listing authority, EDGAR consumer integration, Trade Candidate Builder, KOTH optimizer, Alpaca execution bridge, Synthetic Triple-Barrier Manager runtime, shadow execution, paper execution, and the operator dashboard.

## Pattern System

The long-term design contains 17 alpha patterns across continuous-factor and event-trigger tracks. The codebase currently exposes callable detectors for:

```text
M1, M2, M3, M4, M5, M6, M7, I1, I8
```

M4 is the first production accumulation lane. Detector readiness is not the same thing as tradeability: each pattern still needs feature assembly, candidate construction, liquidity/hazard gates, optimizer integration, execution rules, and validation before it can place orders.

Core pattern examples:

| Pattern | Thesis | Notes |
|---|---|---|
| M4 52-Week High | Right-tail continuation after a split-adjusted close breaks a prior high. | Live accumulation lane. |
| M6 Volatility Compression | Breakout from low-volatility regimes. | Callable detector; production assembly pending. |
| I1 Gap and Go | Confirmed gap continuation. | Callable detector; intraday assembly pending. |
| I8 Opening Range Breakout | Opening-window breakout behavior. | Callable detector; intraday assembly pending. |
| M7 / I10 Pure Technical | ML-derived technical recognition under Reality Check governance. | ML-native layer, not a shortcut around source authority. |

## M4 Live Accumulation Lane

The M4 runner is built around three separate clocks:

| Clock | Meaning |
|---|---|
| Evidence cutoff | The point-in-time regular-session close used for features and context eligibility. |
| Cron fire time | Evening runtime after provider daily bars have finalized. |
| Execution time | The next regular session open, used later for forward-return measurement. |

The canonical runner:

1. Resolves the decision session, evidence session, next execution session, and pinned as-of timestamp.
2. Builds or reads the canonical operating universe.
3. Fetches daily bars only through the evidence-session cutoff.
4. Computes M4 base features including split-adjusted close and prior 52-week high.
5. Enriches a near-breakout superset with source context.
6. Runs the detector over all assembled base inputs.
7. Persists fired signals and no-signal feature evidence.
8. Reuses frozen context on reruns only when schema, setup identity, and as-of conditions match.
9. Emits a health report covering signal count, duplicate identity, provider/source attempts, context coverage, secret/raw-payload checks, PIT cutoff, and forward-return contamination.

## Signal Context

`signal_context` is explanatory evidence stored beside fired M4 signals. It cannot change detector firing, ranking, execution, or return labels.

Current context categories:

| Category | Provider | Purpose |
|---|---|---|
| Identity | Polygon | CIK/FIGI/reference continuity and security metadata. |
| Short interest / short volume | Polygon | Diagnostic short-pressure context with reporting-lag controls. |
| Corporate actions | Polygon | Split/dividend cross-checks and event-date context. |
| News and WIIMs | Polygon, Benzinga | Catalyst density, latest source snippets, URLs, and source attempts. |
| Calendars | Benzinga | Earnings, guidance, ratings, offerings, and dividends. |
| Insider activity | Benzinga | Forms 3/4/5 filing and transaction context. |
| M&A review | Benzinga | High-recall review evidence, not automatic payoff authority. |
| Delisting filings | SEC EDGAR | Official filing authority; adapter committed, consumer wiring next. |

The enrichment layer treats knowledge timestamps and event dates differently. Knowledge timestamps can make a row eligible. Event dates are preserved for inspection but never rescue a row that was not known yet.

## Data Source Authority

Alpha Capital uses data providers according to the role they can actually prove:

| Provider | Role |
|---|---|
| FMP | Universe inputs, historical daily bars, fundamentals, and selected market/reference data. |
| Polygon | Reference identity, corporate-action cross-checks, short-feed context, news, and daily-bar sanity checks. |
| Benzinga | High-recall catalyst/context evidence: news, WIIMs, calendars, insider activity, M&A review. |
| SEC EDGAR | Official filing authority for filings and delisting notices. |
| Nasdaq | Planned authority for halt/listing/daily-list state. |
| Alpaca | Brokerage and execution-time market data. |

Provider boundaries are explicit because wrong authority creates silent data corruption. Benzinga can suggest that something happened; SEC/Nasdaq authority is required where the outcome contract depends on official filing, listing, halt, or delisting status.

## Forward Returns and Path Persistence

Forward measurement is the dependent-variable layer for validation. It must score every simulatable firing, not only trades that eventually pass liquidity or portfolio gates.

Already implemented:

- Entry/exit session planning from persisted `next_execution_session`.
- Mature/immature state handling.
- Split-adjusted open/close price basis.
- Price-finality pending state and provider-revision drift review.
- 15-session daily path persistence in `forward_return_path_rows`.
- `current_forward_path_rows(session, observation)` as the only sanctioned machine-reader API for current forward paths.
- Path preservation rules so a pathless reprice does not wipe last-good audit rows.

Still gated:

- Official survivorship classification from EDGAR/Nasdaq authority.
- Bounded treatment of unresolved mature missing exits in the scoreboard.
- Canonical `run_forward_return` execution.

## Engineering Standards

The project deliberately optimizes for correctness under hostile audit:

- No feature values are silently filled with zero when the value is unknown.
- No canonical forward returns run until source authority is sufficient.
- No provider-latest contamination in PIT evidence.
- No scratch-schema writes to canonical/public tables.
- No secrets, API keys, full DB URLs, or raw provider payloads in feature JSON or logs.
- No production claim from fixtures alone.

Useful checks:

```bash
cd engine
uv run pytest -q
uv run alembic heads
git diff --check
```

## Repository Layout

```text
engine/   Python trading engine: adapters, jobs, feature assembly, detectors, DB models, tests
client/   React/Vite dashboard scaffold
server/   Node/Express API scaffold
docs/     Repo-facing documentation pointers and mockups
```

The complete engineering canon lives outside this code repository in the Alpha Capital vault:

```text
~/Documents/AlphaCapital/
```

Key vault files:

- `Architecture.md` - portfolio construction, runtime contracts, optimizer, execution model.
- `Patterns.md` - canonical 17-pattern roster and exit geometry.
- `Engineering/DataSources.md` - provider authority and adapter completion contract.
- `Engineering/PriceFn.md` - forward-return measurement contract.
- `Engineering/FeatureAssembly.md` - point-in-time feature assembly contract.
- `Engineering/Patterns/` - per-pattern SPEC, DATA, EXPOSURE, EXECUTION, and VALIDATION docs.
- `CODEX.md` - compact recovery context for coding agents.

## What This Demonstrates

For engineering interviews or internships, this project is intended to show:

- Building production-like financial infrastructure from first principles.
- Designing typed adapters around unreliable third-party APIs.
- Writing tests that cover both happy paths and failure-direction health.
- Managing schema evolution with SQLAlchemy and Alembic.
- Preserving point-in-time semantics and lineage across asynchronous jobs.
- Operating a live cloud job with real database writes and provider calls.
- Separating research claims from executable, auditable system behavior.

The system is not yet trading. That is intentional. The current goal is to accumulate clean live evidence first, then let measured outcomes decide what deserves capital.
