# Alpha Capital

Last materially refreshed: 2026-06-15.

Alpha Capital is backend-first infrastructure for systematic U.S. equities
research, signal measurement, historical replay, and supervised ranking. It is
not a dashboard, not a notebook backtest, and not a single detector with a
broker API bolted on.

The project is large because the job is large: prove what the system knew at
decision time, preserve the evidence that caused a signal to fire, measure what
happened afterward without survivorship bias, and train models only from data
that could have existed at serve time.

```text
Universe reconstruction
  -> provider evidence and lineage
  -> pattern-specific feature assembly
  -> signal registry and feature snapshots
  -> forward return/path measurement
  -> market-path and pre-signal feature stores
  -> clean supervised corpora
  -> Stage-1 ranking models
  -> shadow scores / paper execution / optimizer
```

Broker execution is intentionally downstream. The engine is first collecting,
auditing, and scoring the truth set.

## Read This First

This README is a system map, not the legal source of truth. It is expected to
go stale because the platform is changing quickly. When this file disagrees
with code or data, trust sources in this order:

1. Alembic migrations under `engine/migrations/versions/`
2. SQLAlchemy models in `engine/alpha/db/models.py`
3. Current tests under `engine/tests/`
4. Job entrypoints under `engine/alpha/jobs/`
5. This README

The most stale-prone sections are:

- exact corpus row counts
- current test totals
- production DB size and instance sizing
- pattern research verdicts
- ML metric names while the Stage-1 scoreboard is being hardened
- scratch schema names used for one-off corpus builds

If you change the schema, training contract, corpus rules, or production write
guards, update this file in the same branch or explicitly leave a note in the
task checkpoint.

## Current State

This snapshot is date-bound. It is useful for orientation, not for proving the
latest production state.

| Area | State as of this refresh |
|---|---|
| Runtime | Python engine in `engine/alpha`, SQLAlchemy 2.x, Alembic, pytest, scikit-learn, Postgres/Supabase, provider adapters, and guarded CLI jobs. |
| Database | Supabase/Postgres is the canonical target. SQLite exists for local/unit paths only and is refused by canonical write paths. |
| Production compute | Always-on cloud VM runs scheduled jobs; large historical backfills and ML runs are treated as separate heavy jobs rather than fattening the always-on worker. |
| Live patterns | M4, M1, and M2 have production paths. M3 exists but remains default-off. Intraday I11/I12 are research/corpus/paper lanes, not live broker execution. |
| Historical corpus | M4 has a survivorship-correct historical replay and forward-labeling path. I11 and I12 have durable intraday corpus builders with scratch-schema guards. |
| ML layer | Stage-1 ranker machinery exists: manifest loader, leakage-audited feature selection, purged/embargoed walk-forward CV, GBRT trainer, shadow inference, score persistence, registry identity checks, and fail-closed fallback. |
| Safety | Public writes are guarded; scratch schemas are explicitly bound; direct scratch writes verify `search_path`; ML manifests self-check hashes; intraday feature schemas deny known leaky roles by default. |
| Test baseline | Recent local runs are around 2,500 passing tests, 6 skipped, with one known pre-existing Polygon adapter expected-call mismatch around `adjusted=true`. Re-run locally before relying on this number. |
| Execution | Paper execution infrastructure exists. Live broker execution is intentionally not the current milestone. |

## What This System Is Trying To Prevent

Most trading research fails quietly. The curve looks good because the data was
not point-in-time, delisted names disappeared, a provider error looked like
"no signal", or the model trained on a label hiding inside the feature row.

Alpha Capital is built against those quiet failures.

| Failure mode | Defensive design |
|---|---|
| Lookahead leakage | Decision, evidence, execution, and forward clocks are modeled separately. Feature schemas are audited for forbidden roles and label-like names. |
| Survivorship bias | Historical universe reconstruction includes delisted names and separates listing authority from price providers. |
| Provider ambiguity | Provider attempts, payload hashes, statuses, and lineage rows are persisted. Errors are not silently collapsed into empty data. |
| Duplicate signals | Signal identity hashes and uniqueness constraints protect `signal_registry`. |
| Dirty training rows | Feature vector hashes, schema hashes, missing statuses, finite checks, and manifest pins are verified before training/scoring. |
| Scratch/public contamination | Scratch writers bind schema search paths and fail if a session can resolve to `public` unexpectedly. |
| Misleading ML scoreboard | Stage-1 metrics fail closed on flat predictions, tied cutoffs, tiny invalid decile curves, lumpy weights, and unreliable folds. |
| Over-optimistic pattern promotion | Pattern candidates are expected to survive out-of-time, survivorship, cost, and leakage checks before promotion. |

The standard is not "the detector fired." The standard is "we can prove why it
fired, what data was knowable, and how the later label was computed."

## Repository Layout

```text
alpha-capital/
  README.md                         This system map
  engine/
    alpha/
      assembly/                     Pattern-specific feature assembly
      data/                         FMP, Polygon, Benzinga, SEC, Nasdaq, Alpaca adapters
      db/                           SQLAlchemy models and engine/session guards
      evidence/                     Evidence writer and export helpers
      jobs/                         Production jobs, replay jobs, corpus builders, trainer
      ml/                           Manifest loader, feature selector, CV, inference, exclusions
      patterns/                     Pure detector logic and shared contracts
    migrations/                     Alembic schema history
    tests/                          Unit, integration, guard, replay, and ML tests
    pyproject.toml                  Engine package and dependencies
    .env.example                    Environment template, never real keys
  client/                           Dashboard/frontend scaffold
  server/                           API/server scaffold
  docs/                             Dashboard mockups and support notes
  expansion.md                      Feature expansion notes
  m4_backfill_task_list.md          Historical M4 backfill notes
```

The heart of the repo is `engine/alpha`. The client/server scaffolds are not
the trading system.

## Architecture

### Runtime Spine

The canonical nightly and historical jobs share the same design:

```text
1. Resolve trading calendar and run timestamp.
2. Verify the write target.
3. Build or reuse the universe.
4. Assemble pattern inputs.
5. Run pure detectors.
6. Persist evidence, feature snapshots, lineage, and signals.
7. Measure forward returns when windows mature.
8. Build market-path/pre-signal context.
9. Train or score only through a manifest-bound feature contract.
```

The runtime favors explicit records over implicit assumptions. A null value
with `missing_intraday_bars` is not the same as a null value with
`provider_error`, and neither is the same as a computed zero.

### Main Engine Modules

| Module | Responsibility |
|---|---|
| `alpha.data` | Typed provider clients and parsers. |
| `alpha.db` | Database models, engine/session lifecycle, schema guards, writable target checks. |
| `alpha.evidence` | Evidence persistence, export manifests, lineage helpers. |
| `alpha.patterns` | Pure pattern detectors and detector contracts. |
| `alpha.assembly` | Pattern-specific transformation from provider data to detector inputs. |
| `alpha.jobs` | CLI/job entrypoints, canonical orchestration, replay, corpus building, forward measurement, feature backfill, paper execution, training. |
| `alpha.ml` | Manifest loading, stored-feature selection, leakage audit, cross-validation, inference, security-type exclusions. |

## Data Model

The schema is designed for replayability and auditability, not just storing
"signals".

| Family | Representative tables |
|---|---|
| Evidence and runs | `evidence_jobs`, `evidence_job_runs`, `evidence_datasets`, `agent_export_manifests` |
| Provider lineage | `data_lineage`, payload hashes, source attempts, quality flags |
| Universe and identity | `universe_scans`, `canonical_universe_scans`, `universe_snapshots`, `security_profiles`, `security_identity_snapshots`, `nasdaq_listing_snapshots`, `fmp_delisted_companies` |
| Historical reconstruction | `historical_universe_reconstructions`, replay job metrics, scratch schemas |
| Signals | `signal_registry`, `feature_snapshots`, identity hashes, pattern IDs, status fields |
| Forward measurement | `forward_return_observations`, `forward_return_path_rows`, `forward_context_path_rows`, observation events |
| Feature stores | `market_path_features`, `market_path_pre_signal_contexts`, `market_path_pre_signal_links` |
| ML | `ml_model_registry`, `signal_ml_scores` |
| Intraday corpora | `intraday_event_details` plus confirmed `signal_registry`/`feature_snapshots` rows |
| Paper/execution | `paper_execution_events`, `trade_candidates`, `optimizer_runs`, `order_events`, position/lifecycle tables |
| Pattern-specific stores | M1 earnings/friction tables, M2 insider/sec tables, M3 sector/PIT sector history tables |

Many tables intentionally store both machine-consumable values and audit
metadata. That is not accidental bloat; it is how later training and review
distinguish "unknown", "not applicable", "not yet populated", "provider failed",
and "known clean".

## Pattern Portfolio

Patterns are independent detectors fed by pattern-specific assemblers. A
detector should be understandable without knowing how FMP, Polygon, or SEC data
was fetched.

| Pattern | Thesis | Current posture |
|---|---|---|
| M1 | Post-earnings drift / earnings continuation. | Production path exists. Research overlays and hold/avoid filters are treated as part of the evolving spec. |
| M2 | Insider cluster buying from SEC Form 4s. | Production path exists in SEC-first/SEC-only mode where needed. FMP enrichment is not required for the safe path. |
| M2U | Upside-oriented M2 variant. | Built with the M2 lane. |
| M3 | Point-in-time sector rotation. | Detector and sector-history pipeline exist, default-off until governance and coverage gates are met. |
| M4 | Daily 52-week high breakout. | Original live accumulation lane and historical replay anchor. Full replay/forward-label machinery exists. |
| M5 | Failed breakdown reversal. | Evidence archived; not currently promoted after failed daily/minute-bar research attacks. |
| M6 | Volatility compression breakout. | Detector exists; research status can change, check tests and current task notes. |
| M7 | Pure technical ML scaffold. | Effectively downstream of the ML corpus and ranking layer; blocked on mature clean labels. |
| I1 | Gap-and-go intraday/day-0 continuation. | Spec variants have been researched; do not assume production readiness. |
| I8 | Opening range breakout. | Detector exists; production viability depends on intraday data and execution assumptions. |
| I11 | Intraday 52-week-high breakout, day-0 twin of M4. | Durable corpus builder exists. Minute-bar work showed real but smaller edge than daily proxies; overnight exit research matters. |
| I12 | Capitulation volume bounce. | Durable corpus builder exists and is a key intraday research lane. Clean corpus rebuilds must use current code and scratch schemas before any training copy. |

Pattern status is intentionally conservative. "Code exists" does not mean
"tradable".

## Historical Replay And Corpora

### M4 Historical Replay

The M4 replay path answers:

> What would the live M4 engine have fired on each past date if it had been
> running with the then-knowable universe?

Important properties:

- uses historical universe reconstruction rather than today's universe
- includes delisted names when they were alive at the decision date
- separates common-stock training eligibility from registry preservation
- persists reconstructed signals rather than manufacturing labels in memory
- computes forward observations and path rows as first-class records
- protects reruns with idempotency and identity hashes

The historical M4 corpus has been used as the backbone for forward-return
measurement, market-path features, and initial Stage-1 training work.

### I11 And I12 Historical Corpora

I11 and I12 are intraday corpus builders, not casual backtests. They persist
events into `intraday_event_details`; confirmed trainable events also get
`signal_registry`, `feature_snapshots`, and, for I12 current code,
`forward_return_observations`.

Rules that matter:

- build to a scratch schema first
- bind scratch search path explicitly
- keep `public` canonical data protected
- keep split-basis-mismatch rows out of trainable `signal_registry`
- quarantine full-day/research-only fields under `research_only_leaky`
- preserve delisted names for survivorship correctness
- do not promote a corpus to public until feature JSON, labels, dead-name share,
  duplicates, split-basis outliers, and forward observations have been queried
  and audited

The corpus question is not merely "did the edge survive?" The first question is
"is this clean enough to train on without teaching the model a lie?"

## Stage-1 ML Ranker

The Stage-1 ranker is the supervised ranking layer that sits on top of the
measured corpus. It is intentionally narrow: start with per-pattern ranking and
shadow inference before any allocator is allowed to trust the scores.

### What Exists

| Component | Behavior |
|---|---|
| Manifest loader | Loads frozen ML manifests, checks pinned SHA-256, validates horizon contracts, and runs leakage audit during load. |
| Feature selector | Pulls stored features through explicit locators, preserves train/serve vector parity, hashes schema/vector identity, handles typed missing values, and rejects non-finite stored values. |
| Leakage gate | Fails on forward-path roles, label-like names, snake-case leakage labels, forbidden intraday `signal_session` fields, and dotted leaky paths such as `research_only_leaky.*` or `exit.*`. |
| Training loader | Reads `forward_return_observations`, filters by pattern and manifest horizon, collapses duplicate observations to latest, rejects mixed horizons, drops corrupt/non-finite feature rows. |
| CV | Purged/embargoed walk-forward splits with fold-local ticker-cluster weights. Random K-fold is explicitly forbidden for this use. |
| Model | Initial GBRT path via scikit-learn `HistGradientBoostingRegressor`. This is ranker infrastructure, not a claim that GBRT is final. |
| Registry | `ml_model_registry` stores model identity, manifest hash, feature schema hash, artifact URI, CV metrics, and status. |
| Shadow inference | `signal_ml_scores` stores shadow scores. Inference fails closed to fallback on artifact load errors, registry mismatch, OTD/range failures, non-finite features, non-finite predictions, and predict exceptions. |
| Fallback | Raw-strength fallback is idempotent for `model_id = NULL` and has a null-safe unique index. |

### Scoreboard Contract

The Stage-1 scoreboard has been hardened because misleading model metrics are
dangerous. Current intent:

- flat predictions do not inherit time order
- tied score cutoffs fail closed instead of slicing arbitrary rows
- headline pooled top metrics and IC use fold-normalized prediction percentiles
  to avoid fold score-offset bias
- raw pooled IC is preserved separately under `raw_pooled_*` fields
- top-decile lift is `NaN` when population mean is not positive
- decile curves fail closed when any internal boundary is unreliable
- tiny folds use a top-quintile fallback for top metrics but do not pretend full
  decile curves are reliable
- weighted deciles use whole rows, not fractional copies of one signal
- lumpy/heavy weights either report the actual bucket share or fail closed
- fold aggregates do not silently resurrect failed top/decile metrics

Metric names and exact fail-closed behavior are active work. Check
`engine/tests/test_stage1_ml_ranker.py` and `engine/alpha/jobs/train_model.py`
before interpreting a model registry JSON blob.

### What The ML Layer Is Not

It is not "AI picks stocks." It is a supervised ranker over evidence that the
engine already knew how to collect, label, and audit.

The expected research loop is:

```text
clean corpus
  -> manifest-bound feature set
  -> purged walk-forward ranking
  -> shadow scores
  -> paper allocator
  -> execution analysis
  -> promotion or deletion
```

If the corpus is dirty, no model type can save it.

## Market-Path And Pre-Signal Feature Stores

`market_path_features` stores one row per signal per market session. It is the
post-signal and signal-day path spine: base OHLCV, rich technicals, relative
features, ML context fields, hashes, lineage, status flags, and pattern identity.

`market_path_pre_signal_contexts` and links store setup context before a signal
fires. This exists because the model should learn what the setup looked like
before the event, not just what happened on the signal day.

Feature groups include:

- base EOD OHLCV and dollar volume
- 20/60 day liquidity and volume context
- gap, open-to-close, high/low-from-open, entry-relative returns
- realized volatility, ATR/range, wick/body/candle location
- breakout extension and prior high relationships
- compression/base/trend/momentum windows
- benchmark and sector-relative context
- classic technicals such as RSI, ADX/DMI, Bollinger/Keltner, MACD histogram,
  OBV, accumulation/distribution, CMF, stochastic oscillator
- cross-pattern/catalyst placeholders
- explicit null/status fields for future intraday, spread, short/borrow, float,
  offering/halt, and depth sources

Predictor eligibility depends on the decision clock. A field being present in a
row is not enough to make it trainable.

## Provider Authority

Providers have roles. Convenience is not authority.

| Provider | Role |
|---|---|
| FMP | Universe screening, profiles, daily OHLCV, historical replay bars, some fundamentals. |
| Polygon / Massive | Identity details, corporate actions, daily sanity checks, ticker details, short/float style data where available. |
| Benzinga | Catalyst/news/earnings/ratings/offering context. |
| SEC EDGAR | Official Form 4 authority and filing acceptance timestamps. |
| Nasdaq | Listing, halt, and archive authority paths. |
| Alpaca | Paper trading integration. Live broker execution is not the current promoted path. |

Provider failures are observable states. Jobs should preserve whether data was
missing, provider-failed, parse-failed, validation-failed, or excluded by policy.

## Point-In-Time Discipline

The system uses multiple clocks because financial evidence arrives on multiple
clocks.

| Clock | Meaning |
|---|---|
| Decision date | Session for which the engine is deciding eligibility. |
| Evidence session | Market session whose completed data may be used. |
| PIT cutoff | Timestamp after which filings/news/provider updates are not allowed. |
| Signal timestamp | Time the event would have become observable. |
| Execution session/time | Earliest realistic fill assumption. |
| Forward window | Future sessions used only for labels and diagnostics. |

Examples:

- SEC Form 4 acceptance timestamps must be normalized before comparing to a
  decision cutoff.
- M3 sector membership must use formation-date sector assignment, not current
  sector labels.
- Intraday I11/I12 predictors must be as-of-entry, while full-day research
  fields belong under `research_only_leaky`.
- Forward path values are labels/diagnostics unless a manifest explicitly and
  safely says otherwise.

## Operations And Safety

Production writes are intentionally difficult to do accidentally.

| Guard | Purpose |
|---|---|
| `--live` / confirmation flags | Prevent accidental canonical writes from dry-run commands. |
| Postgres canonical target | Canonical writes require the intended database class; SQLite is for local/unit work. |
| Alembic checks | Jobs verify expected migration heads before writing where appropriate. |
| Scratch schemas | Historical/corpus work should land in named scratch schemas before promotion. |
| Scratch search path binding | Scratch sessions bind `search_path` to scratch-first and assert it before writing. |
| Public-write guard | Jobs refuse `public` unless the path is explicitly intended and confirmed. |
| Idempotent upserts | Replay/backfill/corpus jobs should update/reuse rows rather than duplicate them. |
| Source-gated health | Provider failures are fatal only when they invalidate certified evidence. |
| Non-trading day no-op | Calendar-aware jobs exit safely on weekends/holidays. |

If a scratch job writes to `public`, that is an incident. Treat it as data
contamination until scoped deletes and verification queries prove otherwise.

## Common Entry Points

Run from `engine/` unless noted.

```bash
# Local tests
uv run pytest -q
uv run alembic heads

# Canonical nightly dry run
uv run python -m alpha.jobs.run_nightly_canonical --dry-run

# Universe build
uv run python -m alpha.jobs.run_universe --live --trading-date YYYY-MM-DD

# Daily pattern jobs
uv run python -m alpha.jobs.run_m4_daily --live --run-timestamp YYYY-MM-DDTHH:MM:SS-04:00
uv run python -m alpha.jobs.run_m1_daily --live --run-timestamp YYYY-MM-DDTHH:MM:SS-04:00
uv run python -m alpha.jobs.run_m2_daily --live --run-timestamp YYYY-MM-DDTHH:MM:SS-04:00

# Historical M4 replay / measurement
uv run python -m alpha.jobs.run_historical_m4_replay --schema scratch_schema --create-tables
uv run python -m alpha.jobs.run_forward_return --live --pattern-id M4

# Intraday corpora
uv run python -m alpha.jobs.run_i11_historical_corpus --schema scratch_schema --create-tables
uv run python -m alpha.jobs.run_i12_historical_corpus --schema scratch_schema --create-tables

# Market path and pre-signal context
uv run python -m alpha.jobs.run_market_path_features --schema scratch_schema
uv run python -m alpha.jobs.run_market_path_pre_signal_context --schema scratch_schema

# Stage-1 training
uv run python -m alpha.jobs.run_train_model --manifest path/to/manifest.json --pattern-id M4

# Paper execution
uv run python -m alpha.jobs.run_paper_execution --dry-run
```

Read each job's `--help` before running against anything important. Several
jobs have deliberate public/scratch restrictions.

## Local Setup

```bash
cd engine
cp .env.example .env
uv sync
uv run pytest -q
```

`engine/.env` holds database URLs and provider keys. Never commit real keys.

Useful checks:

```bash
cd engine
uv run pytest -q tests/test_stage1_ml_ranker.py
uv run pytest -q tests/test_writable_schema_guard.py
uv run alembic heads
git diff --check
```

The full suite is the real regression signal, but targeted tests are useful
when working on one subsystem.

## Testing And Audit Culture

The test suite is large because the project is data-critical. Recent themes:

- migration and schema integrity
- public/scratch write guards
- provider parser behavior
- market calendar behavior
- detector/assembler contracts
- signal identity and idempotency
- forward-return label computation
- survivorship and historical universe reconstruction
- market-path feature lineage/status/null handling
- ML manifest and feature leakage gates
- purged/embargoed CV and fold-local weights
- shadow scoring fallback behavior
- paper execution isolation from `signal_registry`
- real Postgres scratch probes
- real provider probes where fixture-only tests are insufficient

The workflow intentionally uses adversarial audits. A change is not accepted
because one happy-path test passed. It is attacked for:

- hidden lookahead
- stale global DB sessions
- scratch/public contamination
- duplicate rows
- non-finite values
- tied or flat model predictions
- lumpy ticker weights
- fold leakage
- dirty labels
- impossible entry/exit assumptions

This is why apparently small ML scoreboard work can take many passes. The
first pass computes the metric; the later passes make sure the metric cannot
lie when the model is weak, flat, tied, underweighted, overweighted, or tested
on a tiny fold.

## Known Operational Principles

1. Mark, do not delete. Training exclusions should not mutate the historical
   signal registry.
2. Scratch first. Promote only after query-backed audits.
3. A clean corpus matters more than a promising edge estimate.
4. Pattern code and corpus code must agree on clocks.
5. Full-day fields in intraday corpora are research-only unless proven
   entry-knowable.
6. A provider profile saying "common stock" is not enough for training
   eligibility when series listings, funds, units, notes, or preferreds are in
   play.
7. Scoreboards should fail closed. `NaN` is better than false confidence.
8. Production paths should be boring; research paths can be expensive and
   isolated.

## What Is Not Done

This repo is infrastructure in motion. Important non-final areas include:

- broker live execution promotion
- full allocator/optimizer promotion
- final I11/I12 corpus promotion decisions
- broader Stage-1 pattern manifests beyond the active pattern set
- complete intraday provider entitlement and execution-quality modeling
- final M3 governance/promotion
- final UI/API productization

That is normal. The engine is being built so these decisions can be made with
evidence rather than hope.

## Engineering Scope

This project demonstrates:

- production Python service design
- SQLAlchemy/Postgres modeling
- Alembic migration governance
- point-in-time financial data modeling
- provider adapter hardening
- survivorship-correct historical replay
- idempotent job orchestration
- scratch/canonical environment safety
- feature-store design
- supervised ranking infrastructure
- adversarial metric validation
- cloud scheduling and health monitoring
- systematic-trading risk discipline

The edge can only be trusted if the data path, labels, and operational controls
are trustworthy. This repository is the machinery for earning that trust.
