# Alpha Capital — Private Architecture

**Decision:** Session 25 (2026-05-05); document formalized 2026-05-06
**Decision maker:** Jeb Farneth (founder)
**Status:** ENGINEERING IN PROGRESS — scaffold landed Session 26; Hamilton regime filter next
**Ingestion priority:** Any new Claude session working on Alpha Capital should read this file first. For the public SaaS, read `00-Index/Buttonwood.md`.

Related: [[Buttonwood]] · [[§4-Alpha]] · [[Strategic-Concerns]] · [[Universe-Distribution-Memo]] · [[06-Reference/README|Academic Literature]]

---

## The Decision

Buttonwood is two SEPARATE applications in two SEPARATE git repositories:

1. **Buttonwood** (public SaaS) — the equity research platform at `buttonwood1792.com`. Free and $25/month tiers. No Alpha. Located at `~/buttonwood/`. GitHub: `github.com/jebfarneth/buttonwood` (private).
2. **Alpha Capital** (private) — the personal quantitative trading system. No public domain. Founder-only access via Tailscale private network. Executes trades via Alpaca API. Located at `~/alpha-capital/`. GitHub: `github.com/jebfarneth/alpha-capital` (private).

These are independent applications with separate codebases, separate git repositories, separate GitHub remotes, separate frontends, separate backends, separate dependency trees, and separate deployments. They share a Supabase database instance, an FMP API key, and this specification vault — but no source code, no folder hierarchy, and no git history.

Alpha Capital does not appear in the public Buttonwood UI (Alpha and Options Scanner tabs removed from TopNav in Session 26 Step 0). It is not marketed and is not accessible to anyone other than the founder. It operates as a private tool that generates trading returns on the founder's personal capital ($192K Schwab brokerage → Alpaca brokerage account).

This is not a deferral of the Alpha subscription product. It is a reframing: build the trading system first, generate a track record, then decide whether to open it to subscribers based on 12+ months of auditable performance data.

---

## Why This Path

### Revenue comparison

The SaaS subscription path requires: Phase 3 backend completion, marketing, user acquisition, conversion optimization, customer support, and compliance review — all before the first dollar of revenue. Break-even at 10-23 paying subscribers. Maximum Alpha subscription revenue at 2,000-user cap: $2.4M/year. Realistic time to first revenue: 12-18 months from today.

The private trading path requires: 4-6 engineering sessions to build the nightly scan pipeline, real-time I-pattern monitor, triple-barrier position tracker, and Alpaca execution bridge. Operating cost: ~$135-195/month. Trading begins within weeks of backend completion. Returns compound immediately on existing capital.

At the §4.8 operational magnitudes for the [$30M, $200M] Alpha pick universe — even at 30-50% of face value — private trading returns exceed SaaS subscription revenue until the subscriber base reaches 60-100+ paying users. Private trading has zero customer acquisition cost and zero time-to-first-revenue beyond the engineering build.

### Precedent

Every major quantitative fund started as a personal or small-team trading operation before taking outside capital. Renaissance (Simons, personal capital 1978 → Medallion 1988). Citadel (Griffin, $265K from Harvard dorm 1987 → fund launch 1990). DE Shaw (small team + personal capital → institutional money). The pattern is universal: prove the system works with your own money, generate a track record, then decide whether to scale via outside capital or subscription revenue.

### Track record generation

The vault's own validation methodology (§4.18) requires 90 days of postmortem data before publishing aggregate statistics. Per-pattern rubric calibration needs 6-12 months of live data (Strategic-Concerns Concern 3). The private trading phase IS the validation phase — it generates the exact dataset that makes any future public launch defensible. After 12 months of private trading, the founder has:

- 500-1,000+ resolved picks with full triple-barrier outcome data
- Per-pattern win rates and realized magnitudes
- Empirical validation of which patterns work in 2026-2027 vs academic baselines
- FIM cost framework calibrated against actual execution quality
- Auditable, self-reported track record including every loss

No competitor has this. The track record becomes the strongest marketing asset for a future SaaS launch or fund raise.

---

## Two-Application Architecture

### Buttonwood (public SaaS)

**Location:** `~/buttonwood/client/` (React 19 frontend) + `~/buttonwood/server/` (Express 5 backend)
**Domain:** `buttonwood1792.com`
**Ports:** client :3000, server :5001

**What it includes:**
- TickerSearch (10 tabs: Overview, Financials, Technicals, Earnings, Insider Activity, Options, Analysis, Bonds, ETFs, Seasonals)
- Screener
- Watchlist
- Earnings Calendar
- Politicians (8 tabs)
- Commodities (3 tabs)
- Portfolio / Ironwood with Plaid integration
- BW Composite Score (§2) everywhere
- Hedge page (pending securities lawyer review per DECISION-hedge-personalization-risk)

**What it does NOT include:**
- Alpha picks (no nightly scan, no pattern detection, no pick lists)
- Alpha Performance Stats calendar
- Options Scanner (deferred — depends on Unusual Whales integration)
- Any reference to Alpha Capital in the public UI (Alpha and Options tabs removed from TopNav in Session 26 Step 0)

**Pricing:** Free tier (full research platform) / $15-25/month paid tier (BW Score, Ironwood, Politicians Portfolios/Leaderboard, Financial Health). Exact price point is a future decision.

**Access:** Standard public web application. Email/password signup via Supabase Auth. Open to anyone.

**Launch timing:** Can ship as soon as Phase 3 wires live data to the existing Phase 2 frontend. The frontend is 80% built (38K lines, 37 components). Backend work is primarily FMP data proxy + Supabase CRUD + Stripe payments.

### Alpha Capital (private)

**Location:** `~/alpha-capital/` (separate codebase, separate git repo)
**Access:** Tailscale private network. No public domain.
**Ports:** client :3001 (Vite), server :5002 (Express)
**Status:** Scaffold landed Session 26. Server boots with /health endpoint. Client boots with Vite. GitHub: `github.com/jebfarneth/alpha-capital` (private).

**Tech stack:**
- Server: Express 4 + TypeScript + Supabase + Alpaca Trade API + Axios + node-cron
- Client: React 18 + Vite + TypeScript + Recharts + Lightweight Charts
- Shared base tsconfig with server and client extending it

**What it includes:**
- Nightly scan pipeline: universe filter ([$30M, $200M], price >= $3, common stock), FMP data pull, pattern detection for 13 of 17 spec patterns currently grounded (M1-M6, I1, I3-I8; remaining 4 — M7, I2, I9, I10 — provisional, pending academic ingestion), composite scoring per §4.13, ranked pick output
- Real-time intraday monitor: FMP Professional WebSocket ($79/month) for I-pattern detection (gaps, halts, opening-range breakouts, news catalysts) — continuous, not capped at 4 refreshes
- Triple-barrier position tracker: daily barrier status computation, state-transition logging, postmortem resolution
- Alpaca execution bridge: limit orders placed directly from the Alpha dashboard
- Performance Stats: calendar interface with full self-reporting (the §4.17 spec, running for founder only)
- Alpha Capital is purpose-built for pattern detection and execution; the founder uses the public Buttonwood SaaS separately for general research and screening

**What changes from the §4-Alpha subscriber-product spec:**
- Capacity constraints (§4.20) become optional recommendations displayed in an "Apply recommendations?" modal at execution time, not hard blocks. With N=1, capacity is not binding.
- The 2,000-user admission gate, Plaid-verified capital ceiling, and per-subscriber P_max computation are preserved in the spec but not enforced. They activate if/when the system opens to additional users.
- Intraday refresh limit (4/day) is removed. The system monitors continuously.
- Pick count limit (10 per surface) is removed. The founder sees all survivors above conviction threshold, not a curated top-10.
- Trailing stop orders are the only hard constraint. These execute automatically via Alpaca's trailing stop order type when T1 is hit.

**Access:** Tailscale mesh VPN (free for personal use). The Alpha Capital backend runs on a cloud VM (initial deployment) or local Mac mini (future migration); the host machine joins the founder's Tailscale network and is reachable only from devices on that network — laptop, phone, future hardware. No public domain, no DNS exposure, no public IP. Devices outside the network cannot discover the service exists. Hardware transitions are seamless: install Tailscale on the new device, sign in with the same account, the dashboard is immediately accessible. Cost: $0.

**Execution — Alpaca IS the broker:**

Alpaca is a registered broker-dealer (FINRA member, SIPC insured). When Alpha connects to Alpaca, the founder executes trades through a real brokerage — not routing to Schwab or Robinhood. Capital sits in an Alpaca brokerage account (funded via ACH transfer from Schwab, 1-3 business days). The Alpha backend calls `POST /v2/orders` with ticker, quantity, side, type (limit/market/trailing_stop), and time-in-force. Commission: $0. No per-trade fees. No platform fees. The API is free. The founder sees everything on the Alpha dashboard — positions, fills, P&L, trailing stop status. No separate brokerage app needed. Alpha Capital IS the brokerage interface; Alpaca is the invisible execution layer underneath.

**Operating cost:**

| Item | Monthly |
|---|---|
| FMP Professional (WebSocket + higher rate limits) | $79 |
| Claude API (scoring + catalyst classification) | $50-100 |
| Supabase Free tier | $0 |
| Cloud VM (DigitalOcean / Hetzner — initial deployment) | $5-15 |
| Tailscale (personal plan) | $0 |
| Alpaca brokerage API | $0 |
| **Total** | **$134-194** |

Future migration to Mac mini at home eliminates the cloud VM line, dropping operating cost to $129-179/month after a one-time hardware spend (~$600 for an M-series Mac mini). Trading gains expected to fund this migration.

### Cost savings vs SaaS launch path

| Item | SaaS launch path | Private Alpha path | Savings |
|---|---|---|---|
| Securities lawyer (pre-launch) | $5,000-15,000 | $0 (no public advisory) | $5K-15K |
| Stripe integration + payment infrastructure | $500-1,000 setup + 2.9%/txn | $0 | $500-1K |
| Marketing / user acquisition | $2,000-10,000 | $0 | $2K-10K |
| Customer support infrastructure | $500-2,000/year | $0 | $500-2K |
| Compliance insurance (E&O) | $3,000-8,000/year | $0 (personal trading, no advisory liability) | $3K-8K |
| AWS scaling for multi-user | $100-500/month | $5-15/month (cloud VM single-user) | $85-495/mo |
| Unusual Whales (Smart Money Score) | $125/month (needed for full BW Score) | $0 (defer — Alpha doesn't need options flow for 13 grounded patterns) | $125/mo |
| **Year 1 total operating** | **$15,000-40,000** | **$1,620-2,340** | **$13K-38K saved** |

### What the two applications share (and don't)

**Shared:**
- Supabase database instance (same Postgres; different tables; Alpha tables enforce founder-only access at the route layer)
- FMP API key (same account; Alpha uses Professional tier features)
- Specification vault (`~/Documents/Buttonwood MD Work/Buttonwood/` — separate git repo)

**NOT shared:**
- Project folder (SaaS at `~/buttonwood/`, Alpha at `~/alpha-capital/` — different directories)
- Git repository (`~/buttonwood/` and `~/alpha-capital/` are independent repos with independent histories)
- GitHub remote (`github.com/jebfarneth/buttonwood` vs `github.com/jebfarneth/alpha-capital`)
- Source code (completely independent `node_modules`, `package.json`, TypeScript configs)
- Express servers (SaaS on :5001, Alpha on :5002 — separate Node processes)
- React frontends (SaaS on :3000 via CRA, Alpha on :3001 via Vite — separate builds)
- Public web presence (SaaS at `buttonwood1792.com`; Alpha has no public domain — accessed only via Tailscale)
- Authentication (SaaS uses Supabase Auth for public signup; Alpha uses Tailscale network membership)
- Deployment (SaaS will deploy to public hosting eventually; Alpha runs on a private cloud VM accessed via Tailscale)

---

## Capital Deployment Framework

### Starting position
- Portfolio: $192K (Schwab brokerage + hypothetical $ amount)
- Deployment: 60% to Alpha strategy ($115K), 40% reserve
- Position sizing: $2,000-3,000 per position (1-1.5% of portfolio)
- Simultaneous positions: 30-60 across both surfaces
- Trailing stops: automatic via Alpaca on T1 hit

### Return expectations (from §4.8 operational magnitudes)

The §4.8 T1/T2 targets are calibrated to observed [$30M, $200M] cap behavior. These are what sub-$200M stocks actually do on pattern-qualifying events — not academic baseline magnitudes.

**Multi-day patterns (M1-M6):** T1 +12-15%, T2 +25-30%, Stop -7 to -8%. At 55-65% win rate per §4.21 priors, expected value per trade: +4.6% to +8.8%. With 8-16 resolved trades/month across 6 patterns.

**Intraday patterns (I1, I3-I8):** T1 +15-30%, T2 +30-80%, Stop -7 to -10%. At 50-58% win rate. Expected value per trade: +8.5%+. I3 (Short Squeeze, T2 +75%) and I4 (Halt and Resume, T2 +80%) are the outlier generators. With continuous monitoring: 4-12 resolved trades/month.

**Blended monthly return range:** 8-15% on deployed capital at §4.8 face value. At 50% of face value (conservative validation assumption): 4-8% monthly.

**Annualized (compounded):**
- Conservative (4%/month): $(1.04)^{12}$ = +60%
- Moderate (8%/month): $(1.08)^{12}$ = +152%
- Aggressive (15%/month): $(1.15)^{12}$ = +435%

These ranges assume patterns perform at or near §4.8 operational magnitudes. Months 1-3 of live trading validate this assumption. If patterns perform below 30% of face value, the system flags for review per §4.19 decay monitoring.

### Constraint framework (recommendations, not hard blocks)

At execution time, the Alpha dashboard displays an "Apply recommendations?" module showing:
- Position size recommendation per §4.20 P_max formula
- Sector concentration warning if >30% in one sector
- Pattern diversity warning if >3 picks from same pattern
- ADV liquidity flag if position exceeds 5% of daily volume

The founder can accept or dismiss each recommendation. The only automatic enforcement: trailing stop orders placed via Alpaca when T1 is touched.

---

## Friends-and-Family Expansion Path

After the system validates (3-6 months of live trading with positive returns):

**Phase 1 (no legal structure needed):** Add 5-8 trusted devices to the founder's Tailscale network, OR expose a Cloudflare-Access-protected subdomain at that point if convenience demands it. They see the same Alpha dashboard. They execute their own trades on their own brokerage accounts. This is sharing research — not managing money, not charging fees. SEC Rule 203(b)(3) exempts advisors with <15 clients who don't hold themselves out publicly.

**Phase 2 (if managing outside capital):** Formal investment club (LLC, $500-1,500 setup) or 3(c)(1) private fund structure ($10K-25K legal setup). Only needed if someone wants the founder to manage their capital directly or if profit-sharing is involved.

---

## SaaS Re-Integration Path

After 12+ months of private trading with auditable track record:

1. Re-register Alpha routes in the public app router
2. Add $100/month Stripe tier
3. Implement §4.20 capacity admission gate (the 15-equation framework is already specified)
4. Implement FOMO reveal mechanic (yesterday's picks shown to free/Tier 1 users)
5. Launch with "12 months of auditable, self-reported performance" as the headline differentiator
6. Securities lawyer engagement ($5K-15K) for the personalization/advisory review

The engineering work for the subscription product is already done at this point — Alpha's backend exists and has been running for a year. The new work is subscriber management, payment flow, and the admission gate. Estimated: 2-3 engineering sessions.

---

## Graduate Admissions Framing

The split creates two distinct portfolio entries:

**Buttonwood — AI-Powered Equity Research Platform**
Full-stack web application (React 19 + Express 5 + Supabase + cloud hosting). 10-category composite scoring engine aggregating 48 signals across fundamentals, technicals, sentiment, and alternative data. 37 pages/tabs of interactive financial intelligence. Real-time data integration across 8 providers. Portfolio monitoring with Plaid integration. Demonstrates: software engineering, system design, API architecture, product thinking.

**Alpha Capital — Quantitative Pattern Detection Engine**
17-pattern breakout detection system grounded in 46 peer-reviewed papers across 17 author corpora. Triple-barrier resolution methodology with trailing-stop state machine. Capacity-aware admission control framework with 15 numbered equations derived from Almgren (2005) impact physics, Korajczyk-Sadka (2004) momentum capacity, and O'Neill-Schmidt-Warren (2016) effective-capacity theory. Personal track record: [N months of auditable, DSR-deflated performance]. Demonstrates: quantitative research, financial engineering, statistical validation, risk management.

**MBA admissions framing.** MBA programs evaluate leadership/initiative, quantitative capability, and a narrative arc explaining why the degree is needed. The Alpha split produces an unusually strong application: a solo founder who built both a full-stack financial analytics platform AND a quantitative trading system grounded in 46 academic papers — before starting the program. This demonstrates entrepreneurial initiative paired with quantitative rigor. The personal track record ("I built a quantitative system that returned X% over N months on my own capital") is the kind of concrete, auditable achievement that stands out against the standard "I was an analyst at [bank] for 3 years" application narrative. For Fordham Gabelli MS Business Analytics (STEM, Fall 2026): both entries are directly relevant — data-intensive software + quantitative research methodology. For MBA programs (Wharton, Booth, Columbia, Stern): the combination of a shipping product + a profitable trading system + a research paper is a differentiated application that few candidates can match.

For potential PhD applications (causal reinforcement learning): Alpha IS a decision system operating in a reflexive environment. The 06-Reference corpus (46 papers) demonstrates PhD-scale literature review capability. The vault methodology (DSR, PBO, purged CV with embargo) demonstrates awareness of validation pitfalls in financial ML.

For a research paper: "I built a quantitative trading system grounded in 46 academic papers and validated it against N months of live returns" is a credible research narrative. The vault is the methodology section. The track record is the results section.

---

## Real-Time Architecture (Corrected from Nightly-First)

The intraday patterns (I1, I4, I6, I8) are where the explosive returns live — +20% to +80% T2 targets — and they require real-time detection, not nightly batch processing. The real-time monitor is built alongside the nightly scanner, not after it.

**Market-hours pipeline (real-time, continuous):**

- **Pre-market (4:00-9:30 AM):** System scans overnight gaps via FMP pre-market data. Identifies I1 (Gap and Go) candidates. Pushes alert to founder's phone before market open.
- **Market open (9:30 AM):** Monitors opening-range formation for I8 candidates. Detects within first 30 minutes. Pushes alert.
- **Continuous (9:30 AM - 4:00 PM):** Monitors for news halts (I4), contract announcements (I6), short squeeze setups (I3), sector sympathy moves (I7). Detects within seconds of FMP WebSocket data arrival. Pushes alert. Founder taps to execute via Alpaca.
- **After close (4:00 PM):** Nightly scan runs for multi-day patterns (M1-M6). These are slower-developing setups where nightly is appropriate — earnings drift develops over days, insider clusters accumulate over weeks.

Intraday patterns run real-time. Multi-day patterns run nightly. Both execute via Alpaca. Both track via triple-barrier position tracker.

**Catalyst detection speed — the closeable gap:**

The single most impactful infrastructure upgrade beyond the base operating cost is catalyst detection latency. For I4 (halt resume) and I6 (contract announcement), the difference between detecting the event in 5 seconds vs 5 minutes determines whether the continuation drift is captured or missed.

- FMP Professional WebSocket: streaming quotes during market hours. Detects price movements and volume spikes in real time. Covers I1 (gap detection), I8 (opening-range breakout), I3 (volume surge on short squeeze). Cost: $79/month (already in operating budget).
- SEC EDGAR real-time feed (free): 8-K filings, Form 4 insider transactions arrive within seconds of filing. Covers I6 (contract announcements) and M2 (insider cluster) signal freshness.
- Dedicated news API (optional upgrade): Benzinga Pro ($99/month) or equivalent provides structured news with ticker tagging faster than FMP's news endpoint. Covers I4 (halt catalyst classification) and I6 (contract announcement detection). This is the one upgrade worth considering beyond the base $135-195/month.

**Perplexity API scope note:** The vault (§8-Data-Sources) originally planned Perplexity API ($40-60/month) for news sentiment sweeps, catalyst detection, and geopolitical assessment. In the split architecture, Perplexity's role belongs primarily to the SaaS dimension — editorial content generation (Buttonwood Recommendation panels, Latest News summaries, Wall Street's Take), conversational AI features, and BW Score sentiment signals. For Alpha specifically, Perplexity is NOT required: pattern detection operates on price/volume/fundamentals data from FMP, not on AI-generated editorial. Claude API handles the scoring rubric evaluation and catalyst classification that Alpha needs. If Perplexity is added later for Alpha, it would serve as a supplementary catalyst detection layer — but FMP Professional + SEC EDGAR + optional Benzinga covers the critical path.

---

## Competitive Position vs Institutional Quant Firms

### What Renaissance / Citadel / Jane Street do differently

| Dimension | Institutional quant | Alpha Capital | Gap closeable? |
|---|---|---|---|
| Latency | ~1-10 microseconds (co-located FPGA) | ~200-500 milliseconds (FMP WebSocket → Alpaca API) | NO — physics gap. But irrelevant in the [$30M, $200M] universe. Nobody is co-located for these names. |
| Universe | Liquid large/mid-cap (top 3,000 names) | $30M-$200M (bottom ~1,000 names) | NOT A GAP — different universe. They cannot operate here profitably at their AUM. Alpha can. |
| Data | Proprietary tick-level feeds, dark pool, satellite, credit card | FMP Professional + free APIs | PARTIALLY CLOSEABLE — add Unusual Whales ($125/mo) for options flow + dark pool. Satellite/credit card data is overkill for this universe. |
| Signal count | Thousands of signals, hundreds of factors | 17 patterns, 48 BW Score signals | CLOSEABLE OVER TIME — self-evolving pattern discovery adds new patterns in Year 2-3. But 17 well-validated patterns in an uncrowded universe may outperform 1,000 noisy signals in a crowded one. |
| Execution | Direct market access, internalized flow, optimal execution algos | Alpaca API, limit orders, trailing stops | PARTIALLY CLOSEABLE — Alpaca uses smart order routing. $2K-3K orders in sub-$200M stocks are invisible; execution quality barely matters because the founder is not moving the market. |
| Capital | $10B+ | $192K | NOT CLOSEABLE — but return PERCENTAGE can be higher precisely because Alpha is small. Medallion's capacity is constrained by its size. Alpha's capacity is unconstrained at this size. |
| Holding period | Seconds to days | Hours to weeks | DIFFERENT, NOT WORSE — longer holding in less-liquid names is the correct strategy for this universe. HFT in sub-$200M stocks is unprofitable because spread eats edge. 1-15 day holds are the right timescale. |

### The structural advantage

Alpha operates in a universe institutional capital literally cannot enter. $30M-$200M cap stocks cannot absorb institutional-scale deployment. A $10B fund putting $50M into a $100M cap stock owns half the company. Alpha putting $3K in is invisible. The inefficiencies that exist in this universe — slower information diffusion, zero analyst coverage, behavioral patterns persisting because nobody is arbitraging them — persist precisely because institutional capital cannot reach them. The edge is structural, not technological. Speed helps at the margin (detecting an I4 halt 30 seconds earlier is better than 5 minutes later) but the fundamental edge is operating where the big players cannot.

---

## Engineering Priority (Build Order)

### Completed

- [x] **Step 0: SaaS cleanup** (Session 26) — Alpha and Options Scanner tabs removed from public Buttonwood TopNav and App.tsx routes. AlphaPage.tsx and mockAlphaPicks.ts preserved as reference material. Commit `f17ee92`.
- [x] **Step 1: Alpha Capital scaffold** (Session 26) — Separate server (Express :5002) and client (Vite React :3001) scaffolded with TypeScript compiling clean, server booting with /health, client booting with Vite. Commit `22a66af` (originally in `~/buttonwood/alpha/` before migration).
- [x] **Step 1.5: Codebase split** (Session 26) — Alpha Capital extracted from the buttonwood repo into its own at `~/alpha-capital/`. Both repos pushed to GitHub as private (`github.com/jebfarneth/buttonwood`, `github.com/jebfarneth/alpha-capital`). Codebase boundary now matches the strategic separation.
- [x] **Hamilton (1989) ingestion** (Session 25 closing) — §10 Regime Multiplier methodology anchor. Markov-switching AR model with Hamilton filter. EXTRACT + SYNTHESIS in `06-Reference/Hamilton/`. Commit in vault repo.

### Next (Sessions 26-29)

2. **Hamilton regime filter implementation** — Two-state Markov-switching filter on SPY daily returns → `regime_state` Supabase table → daily expansion_probability → regime_multiplier for §4.13 step 3. Schema in `06-Reference/Hamilton/SYNTHESIS.md`.

3. **Nightly scan + real-time monitor in parallel** — Nightly: Express route for multi-day patterns (M1-M6) via FMP daily data → 6-step composite pipeline → Supabase `alpha_picks` table. Real-time: FMP Professional WebSocket listener for intraday patterns (I1, I3-I8) → event detection → push notification → Supabase logging. Both share the universe filter and composite scoring infrastructure.

4. **Triple-barrier position tracker** — Cron job: market-close trigger → read current prices for all open positions → check T1/T2/Stop/Time barriers → log state transitions to `alpha_pick_states` table → update resolution status.

5. **Alpaca execution bridge** — On pick approval: limit order at entry zone midpoint via Alpaca API. On T1 hit: trailing stop order (10% trailing for multi-day, 12% for I3). On T2/Stop/Time: close position.

### Near-term (Sessions 30-33)

6. **Alpha Capital dashboard** — Minimal UI: pick list, position tracker, performance calendar, execution controls. Accessed via Tailscale (no public URL). Visual reference at `~/Documents/AlphaCapital/alpha-capital-dashboard-mockup.html` (Session 26 mockup, four-page structure: Performance default landing + Picks + Positions + Patterns). Build only after academic ingestion arc completes per founder direction.

7. **Tailscale network setup** — Install Tailscale on the cloud VM running the Alpha Capital backend, on the founder's MacBook, and on the founder's iPhone. Sign each device in to the same Tailscale account. Backend reachable at a private hostname (e.g. `alpha-server.tail-xyz.ts.net`) only from devices on the network. No route-level auth complexity, no DNS exposure, no public IP.

8. **Optional: Benzinga Pro integration** — Structured news feed for faster catalyst detection on I4 (halt) and I6 (contract announcement). Evaluate whether FMP news endpoint latency is sufficient or whether the $99/month upgrade is justified by captured alpha.

### Ongoing

9. **Academic literature ingestion** — Continue Tier 1C pattern-rubric papers per deferred-ingestion queue. Gatheral (2004) or SVI for Hedge page IV surface. 5-7 additional PDFs over 3-6 months.

10. **Rubric weight calibration** — As postmortem data accumulates (90+ days), refine pattern layer weights from uniform starting values using DSR/PBO validation pipeline (§4.18).

11. **Buttonwood SaaS Phase 3 wiring** — In parallel with Alpha trading, wire live FMP data to the public research platform frontend. This work is independent of Alpha and can proceed on a separate track.

---

## What This Document Replaces

This document does not replace §4-Alpha. The pattern specifications, capacity equations, decay haircuts, validation methodology, and academic grounding in §4-Alpha remain canonical. This document adds the OPERATIONAL FRAMING: how §4-Alpha's specifications are deployed as a private trading system rather than a subscriber product.

Sections of §4-Alpha that change in interpretation (not content):
- §4.4 Tier Gating → does not apply to private system (no tiers, one user)
- §4.14 Intraday Refreshes → uncapped for private system (continuous monitoring replaces 4/day limit)
- §4.15 Pick Selection → all survivors shown, not top-10 curated
- §4.17 Calendar Self-Reporting → runs for founder's personal performance tracking
- §4.20 Capacity Admission → recommendations only, not enforcement (N=1 means zero capacity pressure)

Sections that apply unchanged:
- §4.5-§4.8 (pattern architecture and parameters)
- §4.9-§4.12 (triple-barrier resolution, trailing stops, time barriers)
- §4.13 (composite pipeline)
- §4.16 (cross-pattern deduplication)
- §4.18 (statistical validation discipline)
- §4.19 (edge decay monitoring)
- §4.21 (per-pattern rubric status and decay haircuts)

---

## Cross-References

- [[Buttonwood]] — canonical project overview (needs paragraph on two-application model)
- [[§4-Alpha]] — pattern specifications, capacity equations, validation methodology (unchanged)
- [[Strategic-Concerns]] — Concern 4 (universe rationale), Risk 1 (execution gap — mitigated by Alpaca direct execution)
- [[Universe-Distribution-Memo]] — [$30M, $200M] universe rationale
- [[06-Reference/README]] — 17 corpora / 46 papers academic foundation
- [[Session-25-2026-05-05]] — session where this decision was made
- [[Business-Model-Evolution]] — pricing history (SaaS tiers preserved for public product)
- [[DECISION-hedge-personalization-risk]] — Hedge page legal review (applies to SaaS, not private Alpha)

---

*Strategic decision document. Session 25 (2026-05-05); formalized 2026-05-06. Committed after extensive analysis of revenue paths, capital deployment math, regulatory structure, and graduate admissions framing. Engineering to follow in Sessions 26+.*
