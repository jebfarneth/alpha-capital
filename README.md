# Alpha Capital

A private quantitative trading system for US small-cap equities ($30M-$200M market cap). Designed, specified, and built by one person using agentic AI coding tools.

## What This Is

Alpha Capital detects breakout and momentum patterns in ~800-1,200 small-cap stocks, ranks them through a constrained portfolio optimizer, and executes trades via the Alpaca brokerage API. The system is grounded in 79 peer-reviewed academic papers across 50 research corpora.

This is not a backtesting tool or a dashboard. It is a live trading engine designed to deploy real capital, starting at $1K and scaling through validation.

## Architecture

The system uses a factor-model methodology for return prediction and portfolio construction:

- **17 pattern detectors** spanning multi-day momentum (M-track) and intraday breakout (I-track) strategies, each anchored to specific academic literature
- **Evidence spine** — 18-table Postgres schema capturing every signal, candidate, order, position, and validation decision with full point-in-time lineage
- **King-of-the-hill optimizer** — all implemented patterns compete for capital allocation; validation affects confidence weighting, not pattern admission
- **Right-tail-convex exit geometry** — 20/20/60 tranche structure with trailing stops and convexity locks designed to capture explosive microcap moves
- **Shadow + real validation tracks** — statistical validation (Fama-MacBeth + Newey-West HAC) runs continuously against shadow positions; real positions are monitored from day one

## Academic Foundation

| Pattern | Academic Anchor | Key Finding |
|---|---|---|
| M4 52-Week High | Jegadeesh & Titman (1993, 2001) | 1.10%/month momentum premium persists post-publication |
| M6 Vol Compression | Garman-Klass (1980), Lo-Mamaysky-Wang (2000) | Breakouts from GK-measured low-vol regimes produce directional continuation |
| I1 Gap and Go | Lou, Polk & Skouras (2019) | +3.47%/month overnight alpha (t=16.83); confirmation gate filters reversals |
| I8 Opening Range | Heston, Korajczyk & Sadka (2010) | Opening half-hour = ~6x mid-day predictability |

Plus 13 additional patterns covering earnings drift, insider clusters, sector rotation, short squeezes, volatility expansion, FDA catalysts, and more.

## Tech Stack

- **Engine:** Python 3.9 — pattern detectors, data adapters, evidence capture, job orchestration
- **Database:** Supabase/Postgres (production), SQLite (tests)
- **Broker:** Alpaca Trade API — paper and live execution
- **Market Data:** FMP Ultimate (historical/fundamental), Alpaca (live intraday), Polygon (short interest)
- **Dashboard:** React/TypeScript (planned, backend-first approach)
- **Network:** Tailscale mesh VPN — no public exposure

## Specification System

The `/Documents/AlphaCapital/` vault contains the engineering specification for every component:

- `Architecture.md` — portfolio construction, sizing, optimizer, framework contracts
- `Patterns.md` — canonical 17-pattern roster with exit geometry
- `Validation.md` — two-track statistical validation framework
- `Engineering/Patterns/` — per-pattern SPEC, EXPOSURE, DATA, EXECUTION, VALIDATION contracts
- `Engineering/Validation/` — evidence capture, schema, validation job contracts
- `Engineering/RuntimeLayerStack.md` — runtime layer architecture
- `Agentic Engineering.md` — 31-step build sequence for agentic coding recovery

Each pattern detector is implemented strictly from its vault specification. The vault is the source of truth; the code follows it.

## Current Status

**Detector layer (Layer 1 of 9):**
- M4 52-Week High Breakout — complete (10/10)
- M6 Volatility-Compression Breakout — complete (9/10)
- I1 Gap and Go — complete (10/10)
- I8 Opening Range Breakout — complete (10/10)
- 13 patterns remaining

**Shared infrastructure:**
- Evidence spine (18 tables) — complete
- Data adapters (FMP, Alpaca, Polygon) — complete
- Job orchestration with evidence-backed runner — complete
- Universe builder — complete
- Shared detector guards (universe, data confidence, quotes, timestamps) — complete

**Not yet built:** Trade Candidate Builder, KOTH optimizer, execution layer, STBM exit manager, shadow execution, validation jobs, performance telemetry, dashboard.

**Test suite:** 328+ tests passing.

## Development

```
cd engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Environment variables (see `engine/.env.example`):
```
DATABASE_URL=sqlite:///alpha_capital.db
FMP_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
POLYGON_API_KEY=
```

No secrets are stored in this repository. No API keys are required to run tests.

## Repository Hygiene

- Private repository, public-ready at any time
- No secrets, credentials, account IDs, or personal data in git history
- `.env.example` documents required variables without values
- All tests run against SQLite fixtures with no network calls
- Python artifacts, databases, and caches excluded via `.gitignore`
