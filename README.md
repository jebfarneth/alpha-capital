# Alpha Capital

**Autonomous alpha-generation infrastructure for US small-cap equities.**

Alpha Capital is a return-seeking systematic trading engine. The goal is not to publish research, display charts, or collect data for its own sake. The goal is to compound capital: identify repeatable small-cap dislocations, freeze the exact evidence available at the decision time, measure what happened afterward without survivorship bias, and allocate to the edges that actually pay. The end state is a self-improving loop — measured forward returns become the training signal that tells the optimizer, and a machine-learning detection layer, which patterns predict gains, so the edge sharpens as the live signal tape grows and capital concentrates on what works.

The system is built like production infrastructure because the expensive failures in trading are usually silent: a feature leaks tomorrow's filing, a delisting disappears from the sample, a provider returns a partial window that looks like a clean empty result, or a backtest quietly uses data that would not have existed at the time. Alpha Capital is engineered to make those failures visible before capital depends on them.

This is a solo-built, backend-first Python/Postgres trading stack with live unattended signal accumulation already running.

---

## Current Milestone

As of May 2026, the first alpha lane is live in autonomous capture.

| Area | Current state |
|---|---|
| **Live alpha lane** | M4 52-week-high breakout accumulation runs unattended after each trading session. |
| **Universe** | Roughly 671 US-listed small-cap common stocks, rebuilt point-in-time from live provider data. |
| **Signal rate** | About 27-29 M4 signals per live session in the current market regime. |
| **Evidence model** | Fired signals, no-signal feature snapshots, provider attempts, lineage, job runs, forward observations, and daily forward paths are persisted in Postgres. |
| **Forward measurement** | 15-session path persistence is built; official scoring is gated on survivorship authority so delistings and missing exits are not misclassified. |
| **Survivorship authority** | SEC EDGAR adapter is built and hardened; Nasdaq listing/halt authority is the next build. |
| **Operations** | Google Cloud VM + Supabase/Postgres + Healthchecks.io dead-man monitoring. |
| **Regression baseline** | 1,718 Python tests passing; single Alembic migration head. |

The live system is currently in the signal accumulation and outcome-measurement phase. Broker execution is intentionally downstream of validation: the engine is building the production signal tape and forward-return truth set that will decide what deserves capital.

---

## Production Footprint

The canonical M4 runner is deployed on Google Cloud Platform, not just run from a laptop.

| Component | Detail |
|---|---|
| **Compute** | Google Compute Engine `e2-standard-2` VM in `us-east4`, Debian 12, timezone `America/New_York`. |
| **Database** | Supabase/Postgres canonical database, migrated through Alembic. |
| **Scheduler** | Cron at `6:15 PM ET` on weekdays, after provider EOD finalization. |
| **Monitor** | Healthchecks.io dead-man ping around the canonical runner. |
| **Logs** | Persistent runner logs under the engine log path on the VM. |
| **Data egress** | Verified from the VM to Supabase, FMP, Polygon, Benzinga, and SEC EDGAR. |
| **Safety gates** | Non-trading days exit as clean no-ops before DB/provider work; live mode refuses unsafe schema/SQLite targets; Alembic head is checked before canonical writes. |

The cloud job does not rely on calendar intelligence inside cron. The runner owns the trading-calendar decision: if the market is closed, it exits cleanly. If a real health check fails, it exits non-zero and the monitoring path can page.

---

## What It Actually Does

Alpha Capital implements the operating spine of a quant trading system:

```text
Universe -> Feature Assembly -> Pattern Detection -> Signal Registry
         -> Forward Measurement -> Candidate Builder -> Optimizer
         -> Execution -> Exit Manager -> Validation
```

Built today:

| Layer | State |
|---|---|
| **Evidence spine** | SQLAlchemy/Postgres schema, Alembic migrations, job runs, data lineage, universe scans, canonical pointers, feature snapshots, signal registry, forward-return observations, events, and path rows. |
| **Universe builder** | Live FMP screener/profile construction with security-type controls, market-cap bounds, price floors, exclusion telemetry, and Polygon identity enrichment. |
| **Adapters** | Typed FMP, Polygon, Benzinga, Alpaca, and SEC EDGAR adapters. Every provider call returns data plus lineage, source authority, quality flags, and structured errors. |
| **Detector framework** | Callable detector registry with orchestration for feature snapshots, no-signal evidence, signal identity hashes, lineage IDs, and duplicate protection. |
| **M4 lane** | Live 52-week-high breakout feature assembly, signal-context enrichment, canonical daily runner, rerun/idempotency guard, and health report. |
| **Forward returns** | Mature/immature state handling, split-adjusted entry/exit basis, price-finality review, provider-revision review, 15-session daily path persistence, and a sanctioned current-path reader. |
| **Forward-context panel** | Per-signal daily provider-context capture across the 15-session window, with dead-provider quarantine that pages on degradation while preserving unwritten slots for healthy recapture. |
| **Operations** | Google Cloud production runner, Supabase canonical target, Healthchecks.io monitoring, scratch/live safety gates, full regression suite. |

Planned next:

| Layer | Next work |
|---|---|
| **Survivorship clock** | Wire EDGAR into forward-return classification and build Nasdaq listing/halt authority. |
| **Candidate Builder** | Convert validated signals into trade candidates with liquidity, borrow, concentration, and hazard gates. |
| **Optimizer** | Rank and allocate across patterns using measured forward-return distributions. |
| **Execution** | Alpaca execution bridge with order-state, retry, kill-switch, and fill reconciliation. |
| **Exit Manager** | Synthetic Triple-Barrier logic over real fills and live risk state. |

---

## M4: First Live Alpha Lane

M4 targets right-tail continuation after a split-adjusted close breaks a prior 52-week high. This is the first detector promoted from code to unattended live accumulation.

Each canonical run:

1. Resolves the decision session, evidence session, next execution session, and pinned point-in-time cutoff.
2. Builds the live small-cap universe for that session.
3. Fetches daily bars only through the evidence-session close.
4. Computes split-adjusted close, prior high, breakout extension, and M4 exposure.
5. Enriches a near-breakout superset with frozen diagnostic context.
6. Runs the detector and persists fired signals.
7. Persists no-signal feature evidence so non-firings are auditable too.
8. Reruns the assembly path to prove idempotency and context reuse.
9. Emits a health report that can fail in both directions: false-healthy and false-unhealthy.

The system treats a clean no-signal day, a provider-error day, a duplicate-signal day, and a forward-return-contaminated day as different operational states. That matters because a trading system should not just produce signals; it should know when its own evidence is untrustworthy.

---

## Pattern Roster

Seventeen patterns across two tracks, each grounded in peer-reviewed literature. M4 is live; the rest are staged behind it — eight more already have callable detectors, eight are specified and queued. Detector readiness is deliberately *not* tradeability: a pattern earns capital only after point-in-time feature assembly, forward-return validation, and optimizer integration.

**Continuous-factor track** — each pattern is a continuous factor exposure computable for every name every session; the optimizer blends them into one expected-return estimate.

| Pattern | Thesis | Status |
|---|---|---|
| M4 — 52-Week High Breakout | Right-tail continuation after a split-adjusted close breaks a prior 52-week high. | **Live** |
| M1 — Post-Earnings Drift | Earnings-surprise momentum (SUE × decay) through the post-announcement drift window. | Detector built |
| M2 — Insider Cluster | Opportunistic insider intensity × cluster size off Form 4 filings. | Detector built |
| M3 — Sector Rotation | Sector relative-strength rank vs. the universe median. | Detector built |
| M5 — Failed Breakdown Reversal | Short-term reversal off a failed breakdown. | Detector built |
| M6 — Volatility-Compression Breakout | Breakout magnitude after a Garman-Klass low-volatility regime. | Detector built |
| M7 — Pure Technical (ML) | ML-derived multi-day pattern recognition under bootstrap Reality-Check governance. | Detector built |
| I1 — Gap and Go | Overnight-gap continuation gated on intraday confirmation. | Detector built |
| I8 — Opening Range Breakout | Opening-half-hour predictability (Heston-Korajczyk-Sadka). | Detector built |
| I3 — Short Squeeze | Short interest + days-to-cover + loan fee + social intensity. | Spec |
| I5 — Earnings Whisper | Retail-sentiment spike z-score around earnings. | Spec |
| I7 — News Sympathy | Joint-news attention contagion, fast decay. | Spec |
| I9 — ATR Expansion Breakout | Volatility-expansion breakout (ATR / Garman-Klass). | Spec |
| I10 — Pure Technical Intraday (ML) | M7's recognition at intraday frequency. | Spec |

**Event-trigger overlay track** — discrete event-driven; contributes an expected-return overlay only inside its window, then decays to nothing.

| Pattern | Thesis | Status |
|---|---|---|
| I2 — FDA Catalyst | Overlay on FDA regulatory action (PDUFA decision, approval/rejection). | Spec |
| I4 — Halt and Resume | News-halt positive resumption (structurally capped). | Spec |
| I6 — Contract Announcement | Overlay on material contract / partnership announcements. | Spec |

---

## Signal Context

Signal context is frozen beside the feature snapshot. It explains the environment around the signal without being allowed to change the detector result.

| Category | Provider | Purpose |
|---|---|---|
| Identity | Polygon | CIK, FIGI, reference continuity, listing metadata. |
| Short interest / short volume | Polygon | Short-pressure context with explicit reporting lag. |
| Corporate actions | Polygon | Split/dividend cross-checks and event context. |
| News and WIIMs | Polygon, Benzinga | Catalyst density, article counts, source attempts, latest snippets. |
| Calendars | Benzinga | Earnings, guidance, ratings, offerings, dividends. |
| Insider activity | Benzinga | Forms 3/4/5 filing and transaction context. |
| M&A context | Benzinga | High-recall review evidence, never automatic authority. |
| Delisting filings | SEC EDGAR | Official filing authority for Form 25 / 25-NSE evidence. |

Every context category records source attempts. A provider error is not hidden inside an empty count.

---

## Why This Is Hard

The core engineering problem is not writing a detector. It is proving the detector and the outcome measurement are not contaminated.

### Three clocks, never conflated

| Clock | Meaning |
|---|---|
| **Evidence cutoff** | The regular-session close used for feature eligibility. Pre/post-market prints do not feed the breakout definition. |
| **Cron fire time** | Evening runtime after FMP EOD bars are finalized and stable. |
| **Execution time** | Next session open, the earliest realistic fill point for the signal. |

### Point-in-time filings

SEC EDGAR `acceptanceDateTime` looks like UTC but functions as Eastern wall-clock in the submissions feed. The adapter strips the misleading offset, localizes to `America/New_York`, converts to UTC, and applies the PIT cutoff from the converted knowledge timestamp. That closes a real look-ahead trap: filings accepted after the close cannot enter the evidence set just because the provider stamped a confusing suffix.

### Fail-loud survivorship

EDGAR's live ticker map can drop the exact companies that matter most for survivorship. A ticker-only lookup can therefore return a clean empty result for a delisted name. Alpha Capital carries CIK identity forward from the signal and treats unresolved ticker-only survivorship as an error state, not "no event."

### Completeness beats optimism

If an EDGAR submissions window is truncated, the adapter returns `incomplete_window`. It never converts "I could not see the whole window" into "no delisting found." That design prevents false survivorship, one of the most damaging ways to overstate returns.

### Authority boundaries

Benzinga can provide catalyst evidence. Polygon can provide identity continuity and corporate-action cross-checks. SEC EDGAR can prove filings. Nasdaq will prove listing and halt state. The code does not treat those authorities as interchangeable.

---

## Forward Returns

Forward returns are the dependent variable. They decide whether an alpha is real.

The forward-return system is designed to score every simulatable firing, not just the trades that later look convenient. It already supports:

- entry and exit session planning from persisted signal metadata
- split-adjusted entry and exit price basis
- mature vs. immature state transitions
- price-finality pending states
- provider-revision drift review
- 15-session daily path persistence, one row per post-signal session
- synthetic survivorship-exit rows with explicit flags
- current-path reader that refuses stale rows unless the observation is current and computed

Canonical `run_forward_return` is intentionally still gated. The first live M4 cohort matures only after enough trading sessions pass, and scoring it without EDGAR + Nasdaq survivorship authority would be worse than waiting.

---

## The Learning Loop

Forward returns are not just a scoreboard — they are the training signal. Each factor's premium, meaning how much a given pattern actually pays, is designed to be estimated from realized forward-return distributions, so the optimizer allocates to what the live tape proves rather than to priors. Two detectors (M7, and the planned I10) are themselves machine-learning pattern recognizers governed by a bootstrap Reality Check. As the signal tape grows, the estimates sharpen, weak patterns get starved, and capital concentrates on what is measurably paying — a compounding loop in which the edge improves with data.

This loop is the point of the whole system, and it is exactly why the unglamorous parts come first. It stays gated until the forward-return clock opens on honest survivorship authority, because learning from corrupt labels is worse than not learning at all. A single misclassified delisting poisons both the return estimate and everything the model would learn from it — so EDGAR and Nasdaq survivorship authority are built before the ML allocation goes live, not after.

---

## Data Source Authority

| Provider | Role in the system |
|---|---|
| **FMP** | Live universe inputs, historical daily bars, fundamentals, selected reference data. |
| **Polygon** | CIK/FIGI identity, corporate-action cross-checks, short feeds, news, daily-bar sanity checks. |
| **Benzinga** | High-recall catalyst evidence: news, WIIMs, calendars, insider activity, M&A review. |
| **SEC EDGAR** | Official filing authority, including Form 25 / 25-NSE delisting notices. |
| **Nasdaq** | Planned official source for listing, halt, and daily-list state. |
| **Alpaca** | Planned brokerage execution and execution-time market data bridge. |

The system is deliberately multi-source because trading errors often come from letting a convenient provider answer an authority question it cannot actually prove.

---

## Engineering Rigor

This project is built to survive hostile review.

Current baseline:

- **1,718 Python tests passing**
- **single Alembic head**
- SQLite test coverage with Postgres-targeted schema discipline
- scratch/live runner gates
- no production writes from scratch mode
- secret/raw-payload scans in canonical health reporting
- live-read probes for provider behavior where fixtures are not enough
- adversarial audits for both false-positive and false-negative health outcomes

Examples of defects the test/audit process caught and closed:

- broad secret scanning that could block a clean run on benign text
- failed same-date runs contaminating health reports
- failed metrics overriding successful metrics
- EDGAR ticker-only delisting misses
- EDGAR pagination false-empty windows
- CIK normalization that could turn malformed identity strings into valid-looking CIKs
- forward-path rows being wiped by pathless repricing
- stale ORM objects serving stale forward paths
- Polygon daily volume parsing rejecting valid float volumes
- dead provider keys silently writing poisoned forward-context rows that consumed unrecoverable capture slots

The standard is not "tests pass." The standard is: try to break it, then prove why it still holds.

---

## What This Demonstrates

This repository is intentionally backend-heavy. It is meant to show the kind of engineering required to turn a trading idea into a system that can make and measure capital-allocation decisions.

It demonstrates:

- production Python systems design
- SQLAlchemy/Postgres data modeling
- Alembic migration discipline
- typed provider adapters with lineage and error contracts
- point-in-time data engineering
- trading-calendar correctness
- survivorship-bias control
- cloud deployment on Google Cloud Platform
- operational monitoring with Healthchecks.io
- adversarial test design
- autonomous batch orchestration
- security hygiene around secrets and raw provider payloads

The alpha thesis matters, but the infrastructure is the moat: if the measurement is wrong, the return estimate is fantasy.

---

## Verify Locally

```bash
cd engine
uv run pytest -q
uv run alembic heads
git diff --check
```

---

## Repository Layout

```text
engine/   Python trading engine: adapters, jobs, feature assembly, detectors, DB models, tests
client/   React/Vite dashboard scaffold
server/   Node/Express API scaffold
docs/     Repo-facing documentation pointers
```

---

## Roadmap

1. Wire SEC EDGAR into forward-return survivorship classification.
2. Build Nasdaq listing/halt/daily-list authority.
3. Open the forward-return clock and score the first matured M4 cohort.
4. Build the scoreboard over full 15-session paths, not just endpoint returns.
5. Convert validated signals into trade candidates.
6. Add optimizer, portfolio constraints, and execution bridge.
7. Move from live accumulation to shadow/paper trading, then capital deployment only after the measured edge justifies it.

---

Alpha Capital is a for-profit trading system in construction. The data work, lineage discipline, cloud operations, and audits are not academic overhead; they are what make the eventual return numbers worth trusting.
