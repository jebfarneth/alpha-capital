# Alpha Capital

Alpha Capital is a private systematic trading engine for US micro/small-cap common equities, focused on the $30M-$250M market-cap universe. It is built as a complete quant operating system: point-in-time data ingestion, feature assembly, pattern detection, portfolio allocation, execution, exit management, and continuous validation.

The system is designed for real capital from day one, starting around $1K and scaling only as live and shadow evidence justify it. It is not a dashboard-first app, a backtesting toy, or a discretionary alert feed.

## Operating Model

Alpha Capital scans a live operating universe, assembles point-in-time feature inputs, runs a 17-pattern alpha registry, ranks candidates through a constrained optimizer, routes orders through Alpaca, manages exits through a synthetic triple-barrier manager, and records every decision into an evidence spine for validation.

The completed system is designed around four principles:

- **Right-tail capture:** preserve exposure to explosive microcap moves through terminal-tranche-heavy exits, wide trailing stops, and convexity locks.
- **Pattern competition:** every eligible pattern can enter the king-of-the-hill allocator; validation changes confidence and sizing, not whether a pattern is allowed to produce evidence.
- **Point-in-time accountability:** signals, features, data lineage, timestamps, universe membership, candidates, orders, positions, and exits are recorded so every claim can be audited after the fact.
- **Small-AUM pragmatism:** the portfolio model accepts that a $1K-$10K account should optimize for asymmetric opportunity capture before institutional smoothness.

## Pattern System

The production design contains 17 patterns across two tracks.

**Continuous-factor track:** M1, M2, M3, M4, M5, M6, M7, I1, I3, I5, I7, I8, I9, I10. These produce observable factor exposures and are validated with Fama-MacBeth regressions plus Newey-West HAC errors.

**Event-trigger overlay track:** I2, I4, I6. These add event-window expected-return overlays and are validated with event-study CAR methodology.

Core examples:

| Pattern | Thesis | Anchor |
|---|---|---|
| M4 52-Week High | Right-tail-convex continuation after split-adjusted close breaks a prior high | Jegadeesh & Titman |
| M6 Volatility Compression | Breakout from low-volatility regimes | Garman-Klass; Lo-Mamaysky-Wang |
| I1 Gap and Go | Confirmed gap continuation | Lou, Polk & Skouras |
| I8 Opening Range Breakout | Opening-window predictability | Heston, Korajczyk & Sadka |
| M7 / I10 Pure Technical | ML-derived pattern recognition under Reality Check governance | Cakici-style technical features |

The vault contains the full pattern roster, exit geometry, validation thresholds, and source literature.

## Architecture

The stack is backend-first and evidence-driven:

1. **Universe Builder** - builds the tradable $30M-$250M operating universe with security-type, country, exchange, price, and liquidity controls.
2. **Data Layer** - FMP for historical bars and fundamentals, Polygon for market/reference context, corporate actions, short feeds, and news, Benzinga for catalyst/context feeds, and Alpaca for live intraday and execution-time prices.
3. **Feature Assembly** - converts provider data into pattern-specific `PatternInput` objects with typed missing values, lookahead enforcement, lineage, deterministic feature hashes, and frozen diagnostic context.
4. **Detector Registry** - runs callable pattern detectors against assembled inputs and records both signals and valid no-signal evidence.
5. **Evidence Spine** - Postgres/Supabase schema for jobs, lineage, universe snapshots, feature snapshots, signals, candidates, orders, positions, exits, and validation telemetry.
6. **Trade Candidate Builder** - merges co-firing signals, applies mutexes and vetoes, estimates net edge after costs, and emits auditable trade candidates.
7. **King-of-the-Hill Optimizer** - allocates limited capital to the best candidates subject to AUM tier, reserve, correlation, liquidity, and risk constraints.
8. **Execution Bridge** - routes orders through Alpaca using class-specific order semantics.
9. **Synthetic Triple-Barrier Manager** - owns staged take-profits, stops, trailing stops, time barriers, framework exits, reconciliation, and recovery.
10. **Validation Layer** - maintains shadow and real tracks, forward returns, factor returns, event CARs, pattern weights, confidence haircuts, and decay monitoring.

## Implementation Status

This repository is not a paper design. Several production-path backend pieces are already implemented and tested, while later trading-system layers remain intentionally blocked until the evidence spine proves itself.

| Area | Current state |
|---|---|
| Database / evidence spine | SQLAlchemy models, Alembic migrations, job/run records, lineage rows, universe scans, feature snapshots, signal registry, and forward-return observation tables exist. |
| Universe build | Canonical universe construction is active with security-type enrichment, Polygon identity enrichment, inclusion/exclusion telemetry, and scratch-schema rehearsal support. |
| Data adapters | FMP, Polygon, Benzinga, and Alpaca adapters exist behind typed contracts and lineage metadata. Polygon/Benzinga adapters have the most recent hardening attention. |
| Detector orchestration | Callable detectors persist signals and no-signal feature evidence through a shared orchestration path with feature hashes, lineage IDs, and dedupe rules. |
| Implemented detector path | M4 daily 52-week-high breakout is the live production-path pattern under accumulation. Other pattern detectors/assemblers exist in varying stages but are not all on the canonical daily accumulation path yet. |
| Signal context | M4 has pre-detection `signal_context` enrichment for near-breakout candidates, with fired signals carrying frozen source context into `feature_json`. Pattern-wide enrichment is the design target, not fully rolled out. |
| Forward returns | Schema and job scaffolding exist, but canonical forward-return computation remains blocked until official EDGAR/Nasdaq authority paths are built. |
| Trading layers | Trade Candidate Builder, KOTH optimizer, Alpaca execution bridge, STBM exits, and operator dashboard remain downstream build phases. |

## Current Build Frontier

The repository is being built toward the architecture above. The active backend now includes the evidence spine, canonical universe construction, Polygon identity enrichment, callable detector orchestration, the M4 daily production assembly path, and M4 `signal_context` enrichment.

M4 daily wiring is past the first production slice: the job resolves market sessions, assembles split-adjusted close features, computes prior-session 52-week highs, enforces short-history floors, persists feature/no-signal/signal evidence, deduplicates signals, and carries lineage through detector orchestration. M4 signal context is attached before detection for a near-breakout superset so fired M4 signals keep frozen source context without enriching the full universe.

Current provider-context coverage includes Polygon identity, short interest, short volume, splits, dividends, and news, plus Benzinga news/WIIMs, earnings, guidance, ratings, offerings, dividends, insider filings/transactions, and M&A review context. These adapters are treated as source/context evidence with point-in-time guards, lineage hashes, error sanitization, payload-shape checks, and bounded feature JSON summaries.

The current frontier is canonical M4 signal accumulation: turning the scratch-proven daily M4 flow into a guarded recurring canonical run with deterministic session clocks, Supabase/Postgres health reporting, frozen context reuse, secret/bloat checks, forward-return guards, and explicit scratch-vs-live write gates. The canonical runner is under adversarial audit before it lands; until then, the committed live-write rehearsal path is the scratch runner.

After canonical M4 accumulation is trusted, the build sequence expands enrichment and accumulation across the remaining 16 patterns, then moves through EDGAR/Nasdaq authority, Trade Candidate Builder, optimizer, execution bridge, STBM exits, validation jobs, and operator dashboard.

## M4 Daily Flow

The M4 path is the first production accumulation lane. Its current shape is:

1. Resolve the U.S. equity decision session, evidence session, and next execution session.
2. Load the canonical operating universe for the decision date.
3. Fetch and assemble split-adjusted daily bars through the evidence-session cutoff.
4. Compute the M4 base features, including completed-session close and prior 52-week high.
5. Attach stable signal identity fields and setup hashes before detection.
6. Select a near-breakout superset for API-rich `signal_context` enrichment.
7. Enrich only that superset, preserving provider attempts, warnings, lineage IDs, and event dates as context.
8. Run the detector over all assembled M4 inputs, not only enriched inputs.
9. Persist fired signals and no-signal feature evidence through detector orchestration.
10. Reuse frozen context on rerun when persisted context matches schema, as-of timestamp, and setup identity.

The near-breakout prefilter is a cost-control mechanism, not a detector. It is currently scoped to the M4 base lane and is guarded by tests so it does not exclude base-lane firings. Fresh/watchlist lane logic requires a separate threshold and price-basis audit before reuse.

## Signal Context

`signal_context` is diagnostic source context attached to feature JSON. It is not allowed to change detector firing, price inputs, KOTH ranking, or forward-return labels. Its purpose is to make future validation explainable: when a signal fired, the evidence record can show what relevant source context was known at the signal cutoff.

Current M4 context categories include:

| Category | Source | Use |
|---|---|---|
| Identity | Polygon | Ticker/security identity, exchange, type, and reference continuity. |
| Short interest | Polygon | Short-interest level, days-to-cover, and replay-safe availability gating. |
| Short volume | Polygon | Recent short-volume context with reporting-lag controls. |
| Corporate actions | Polygon | Splits/dividends as context-only event dates with availability gating. |
| News | Polygon and Benzinga | Article counts, latest titles/URLs, WIIMs, source attempts, and PIT publication checks. |
| Calendars | Benzinga | Earnings, guidance, ratings, offerings, and dividends. |
| Insider activity | Benzinga | Form 3/4/5 filing and transaction context, including routine/discretionary classification inputs. |
| M&A review | Benzinga | Review context only; production payoff authority still requires official identity continuity. |

The enrichment layer treats knowledge timestamps and event dates differently. Knowledge/availability timestamps can make rows eligible. Event dates are preserved for inspection but do not prove point-in-time availability. Rows with unavailable or future knowledge timestamps are excluded from PIT counts rather than rescued with event dates.

Feature JSON intentionally stores summaries, counts, statuses, latest source snippets, and source-attempt telemetry. Raw provider payloads belong in lineage tables, not in feature JSON.

## Data Source Posture

Alpha Capital uses providers according to authority level:

| Provider | Role |
|---|---|
| FMP | Historical bars and fundamental/universe inputs. |
| Polygon | Identity/reference data, market context, short feeds, news, and corporate-action cross-checks. |
| Benzinga | Catalyst/context data: news, WIIMs, calendars, insider activity, dividends, offerings, and M&A review rows. |
| Alpaca | Live brokerage/execution-time market data and eventual order routing. |
| EDGAR/Nasdaq | Not yet implemented; required before official corporate-action/listing-status authority and canonical forward returns. |

Polygon and Benzinga are being used aggressively as context sources, but they are not treated as official legal authority for every event. For example, Benzinga M&A data is useful review evidence, while forward-return resolution still waits for EDGAR/Nasdaq-grade authority.

## Hard Stops

Several boundaries are deliberately fenced off:

- **No canonical forward returns yet.** Forward-return tables and tests exist, but official return labeling is blocked until EDGAR/Nasdaq authority paths are implemented and audited.
- **No detector firing changes from context.** `signal_context` is explanatory evidence. Detector inputs and firing thresholds remain owned by pattern code.
- **No full-universe rich enrichment by default.** M4 enriches a near-breakout superset so fired signals have context without recurring full-universe API fanout.
- **No scratch writes to canonical tables.** Scratch rehearsals must use explicit schemas. Canonical writes must use the default search path and explicit confirmation.
- **No hidden secret or raw-payload persistence in feature JSON.** Feature JSON is bounded and diagnostic; raw provider rows belong in lineage storage.
- **No production claim from unit tests alone.** Live scratch reads/writes and adversarial audits are part of the acceptance path.

## Repository Layout

```text
engine/   Python trading engine: data adapters, feature assembly, detectors, jobs, evidence, DB models
client/   React/Vite dashboard scaffold
server/   Node/Express API scaffold
docs/     Repository documentation pointers
```

The complete engineering canon lives outside this repository in the Alpha Capital vault:

```text
~/Documents/AlphaCapital/
```

Important vault files:

- `Architecture.md` - portfolio construction, sizing, optimizer, and runtime contracts
- `Patterns.md` - canonical 17-pattern roster and exit geometry
- `Validation.md` - shadow/real validation framework
- `Engineering/FeatureAssembly.md` - feature assembly contract
- `Engineering/Patterns/` - per-pattern SPEC, EXPOSURE, DATA, EXECUTION, VALIDATION docs
- `Engineering/RuntimeLayerStack.md` - runtime-layer build sequence
- `CODEX.md` - high-density recovery context for AI coding agents

## Key Engine Entrypoints

The engine is operated through explicit jobs and guarded CLIs rather than ad hoc scripts:

| Command/module | Purpose |
|---|---|
| `alpha.jobs.run_universe` | Build the daily operating universe and canonical universe snapshot. |
| `alpha.jobs.run_m4_daily` | Run the M4 daily assembly and detector-orchestration path from the canonical universe. |
| `alpha.jobs.run_m4_launch_scratch` | Launch-like Supabase/Postgres scratch rehearsal for universe + M4 daily + signal-context coverage. |
| `alpha.jobs.run_forward_return` | Forward-return runner scaffold. It remains blocked for canonical use until official authority paths are ready. |
| `alpha.jobs.detector_orchestration` | Shared persistence path for feature snapshots, signals, no-signal evidence, hashes, and lineage. |

Local unit tests normally run without live keys. Live provider tests and Supabase scratch rehearsals are explicit operator actions because they touch rate limits, external state, and real database schemas.

## Development

Engine setup:

```bash
cd engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uv run pytest -q
```

Full app scaffolds:

```bash
npm run install:all
npm run dev
```

Environment variables are documented in:

```text
engine/.env.example
server/.env.example
```

No secrets are committed. Tests are expected to run without live API keys unless a task explicitly invokes a live-data audit.

Common engine checks:

```bash
cd engine
uv run pytest -q
uv run alembic heads
git diff --check
```

Live-read and live-write rehearsals are run through guarded jobs and scratch schemas first. Canonical writes require explicit confirmation flags and must never use a scratch `ALPHA_DB_SCHEMA`.

## Production Standard

Alpha Capital treats live-read and live-write audits as part of implementation, not afterthoughts. A feature is not production-ready because fixtures pass. It must prove, in scratch schemas and with real provider data, that it preserves point-in-time semantics, lineage, determinism, deduplication, source-attempt explainability, and canonical/public isolation.

That standard is deliberate: this system is meant to trade real money, preserve the right tail, and leave an evidence trail strong enough to debug both profits and mistakes.
