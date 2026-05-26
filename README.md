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

The completed stack is backend-first and evidence-driven:

1. **Universe Builder** - builds the tradable $30M-$250M operating universe with security-type, country, exchange, price, and liquidity controls.
2. **Data Layer** - FMP for historical/fundamental data, Alpaca for live intraday and execution-time prices, and supplemental sources for short interest, filings, catalysts, halts, FDA events, and news.
3. **Feature Assembly** - converts raw provider data into pattern-specific `PatternInput` objects with typed missing values, lookahead enforcement, lineage, and deterministic feature hashes.
4. **Detector Registry** - runs callable pattern detectors against assembled inputs and records both signals and valid no-signal evidence.
5. **Evidence Spine** - Postgres/Supabase schema for jobs, lineage, universe snapshots, feature snapshots, signals, candidates, orders, positions, exits, and validation telemetry.
6. **Trade Candidate Builder** - merges co-firing signals, applies mutexes and vetoes, estimates net edge after costs, and emits auditable trade candidates.
7. **King-of-the-Hill Optimizer** - allocates limited capital to the best candidates subject to AUM tier, reserve, correlation, liquidity, and risk constraints.
8. **Execution Bridge** - routes orders through Alpaca using class-specific order semantics.
9. **Synthetic Triple-Barrier Manager** - owns staged take-profits, stops, trailing stops, time barriers, framework exits, reconciliation, and recovery.
10. **Validation Layer** - maintains shadow and real tracks, forward returns, factor returns, event CARs, pattern weights, confidence haircuts, and decay monitoring.

## Current Build Frontier

The repository is being built toward the architecture above. The backend foundation is active: universe construction, evidence capture, detector orchestration, callable detector contracts, and the first M4 feature-assembly slice are in place. Production readiness is not declared from unit tests alone; each new boundary is validated through live scratch-schema audits before being trusted.

The immediate frontier is M4 daily production wiring: market-calendar-aware session resolution, split-adjusted close feature assembly, strict prior-session 52-week-high computation, hard short-history signal floors, feature persistence, lineage proof, and no-signal deduplication.

After that, the build sequence moves through the remaining feature assemblers, Trade Candidate Builder, optimizer, execution bridge, STBM exits, validation jobs, and operator dashboard.

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

## Production Standard

Alpha Capital treats live-read and live-write audits as part of implementation, not afterthoughts. A feature is not production-ready because fixtures pass. It must prove, in scratch schemas and with real provider data, that it preserves point-in-time semantics, lineage, determinism, deduplication, and canonical/public isolation.

That standard is deliberate: this system is meant to trade real money, preserve the right tail, and leave an evidence trail strong enough to debug both profits and mistakes.
