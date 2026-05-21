# Root Canonical Documents Fresh-Eyes Audit

**Timestamp:** 2026-05-14 07:25:13 EDT  
**Scope:** Architecture.md, Patterns.md, Universe.md, Validation.md, Regression-Checklist.md, SyntheticTripleBarrierManager.md  
**Posture:** Antagonistic, fresh-session, read-only

## Executive Call

No CRITICAL findings surfaced. However, the root canon is **not clean for new pattern drafting** until the HIGH findings below are resolved. The dominant issue is stale root-canon coexistence: older execution, track-type, leverage, data-provider, and validation-authority language remains live next to newer Wave 2.11/STBM/checklist language.

## Summary by Severity

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| HIGH | 6 |
| MEDIUM | 5 |
| LOW | 1 |
| NOTE | 1 |

## Findings

[HIGH] Architecture.md:920 — I-track definition is stale relative to the intended <1d-to-5d track scope  
Operator clarification: I-track means short-horizon positions from less than one day up to 5 days; M-track means 5-20 days. Architecture currently defines all I1-I10 as opening and closing within a single trading session with no overnight hold, while Patterns.md correctly assigns I-track and event-overlay rows 2-5 day time barriers. Regression-Checklist.md:516-520 inherits the stale same-session assumption by saying I-track must not use `time_barrier` or `optimizer_rebalance`, which is wrong for multi-session I-track positions.  
Recommended action: Update Architecture and Regression-Checklist to define I-track as short-horizon/intraday-to-5-day, not same-session only. Preserve the Patterns.md I-row time barriers, then clarify which I-track framework exits apply to any position that can cross sessions.

[HIGH] Architecture.md:694 — Execution Optimization order-routing rules contradict Wave 2.11 class routing  
The older execution section still says time-locked patterns use market orders, high-conviction non-time-locked picks use market orders, lower-conviction rebalancing uses 5-minute midpoint limits with fallback to market, and T1/T2 exits use market orders. Architecture.md:1224-1239 and Regression-Checklist.md:79-85 instead canonicalize Class A midpoint-limit day-valid entries and triple-barrier GTC limit take-profits under STBM. An implementation agent could choose the wrong order type, TIF, and exit primitive from canon alone.  
Recommended action: Replace Architecture.md:694-701 with the Wave 2.11 class-routing and STBM exit-order rules, or explicitly mark the older strategy text as superseded/non-canonical.

[HIGH] Architecture.md:814 — Leverage deployment arithmetic contradicts the stated 90%/120% effective deployment targets  
The canonical formula says `target_gross_exposure = active_leverage_tier * deployable_fraction`; with the V1 75% deployable fraction, 1.5x implies 112.5% gross exposure and 2.0x implies 150%. Architecture.md:1523 instead states 1.5x gives 90% and 2.0x gives 120%. This is a material capital-deployment ambiguity that can cause incorrect sizing and risk.  
Recommended action: Define the actual levered gross-exposure schedule as either formula-driven or table-driven, and reconcile the reserve rule against NAV, equity, and gross notional.

[HIGH] Architecture.md:405 — FMP-only FULL dependency claim conflicts with required external structured sources  
Architecture says I2, I4, and I6 have FULL dependencies satisfied entirely by FMP Ultimate baseline data. The same document later requires I2 FDA Calendar/PDUFA verification and I6 SEC 8-K verification, and Regression-Checklist.md:234-238 preserves FDA.gov and SEC EDGAR as operator decisions. This can cause a builder to omit required non-FMP event verification while still marking the pattern FULL.  
Recommended action: Split "market/fundamental provider" from "structured event verification" and list FDA.gov plus SEC EDGAR as direct FULL dependencies for I2 and I6.

[HIGH] SyntheticTripleBarrierManager.md:158 — T2 fill handling contradicts the state-aware stop ratchet contract  
The core state machine says T2 fill cancels the active stop, submits a replacement GTC stop at the T1 floor, and only lets trailing replace that floor when the trailing level exceeds it. The later T2 handling section instead says to submit the trailing stop immediately and mark `TRAILING_ACTIVE`. That bypasses the required `post_t2_floor` protection state and conflicts with the M4-specific parameter block.  
Recommended action: Rewrite T2 handling so the immediate post-T2 protective order is the T1-floor stop, with trailing activation gated by the computed trailing level exceeding that floor.

[HIGH] Validation.md:195 — Validation weight gates conflict with Architecture's pattern-weight authority boundary  
Validation says PBO/DSR results directly produce full, half, or zero optimizer weight. Architecture.md:463 says validation diagnostics do not directly mutate pattern weights; weight changes flow through execution_capture or explicit operator override. If these are meant to be different layers, the canon does not name the separate variable, which creates governance ambiguity for live allocation.  
Recommended action: Introduce an explicit `validation_weight_multiplier` or `optimizer_eligibility_gate` separate from `pattern_weights.current_weight`, then update both documents to use that boundary consistently.

[MEDIUM] Architecture.md:1549 — Return-expectation section is stale relative to earlier Architecture projections  
The closing Return Expectations section cites 11-13% monthly unleveraged, 16-23% leveraged, and year-1 50th percentile $4k-$6k. Earlier Architecture text around trade-flow calibration describes an updated aggregate projection and year-1 50th percentile $6k-$9k. This is not an implementation break, but it undermines which projection is canonical.  
Recommended action: Keep one canonical projection block and replace later shorthand with a pointer to that block only.

[MEDIUM] Architecture.md:28 — FMP Premium vs FMP Ultimate naming is inconsistent  
The root documents alternate between FMP Premium and FMP Ultimate for the same V1 primary provider. Regression-Checklist.md also mixes "FMP Premium" in the adapter invariant with "FMP Ultimate inherited" in pattern invariants. Since provider tier determines available endpoints and cost, this is more than cosmetic naming drift.  
Recommended action: Choose the exact subscription/tier label and normalize Architecture, Regression-Checklist, and pattern dependency wording.

[MEDIUM] Universe.md:19 — The $30M floor remains empirically provisional, but Architecture treats it as production-canonical  
Universe.md explicitly says ADV percentile validation is deferred and the $30M floor is directional pending FMP extraction. Architecture and downstream pattern docs use $30M-$200M as live universe canon. This is acceptable as a thesis, but the missing empirical validation gate should be explicit before capital deployment.  
Recommended action: Add a launch gate requiring FMP-derived ADV/spread validation for the $30M-$50M bucket, or mark the floor as thesis-conditional everywhere it drives production sizing.

[MEDIUM] Architecture.md:1085 — trade_candidates does not persist the cost component used to compute net_expected_edge  
Architecture defines `net_expected_edge = combined_expected_edge - expected_round_trip_cost`, but the canonical `trade_candidates` table stores `combined_expected_edge` and `net_expected_edge` without storing `expected_round_trip_cost`. That makes cost arithmetic harder to audit after the fact and forces reconstruction from external context.  
Recommended action: Either add a canonical `expected_round_trip_cost` field or specify that it must be persisted inside a named `source_features` key with stable units and provenance.

[MEDIUM] Architecture.md:1232 — STBM scope is M-track only while short-horizon I-track rows also use triple-barrier exits  
Architecture says STBM owns lifecycle management for multi-tranche M-track patterns M3-M6. Under the clarified track taxonomy, I-track can hold up to 5 days, and Patterns.md assigns multi-tranche T1/T2/T3 exits, hard stops, time barriers, and trailing stops to I1/I3/I5/I7/I8/I9/I10 and I2/I4/I6 overlays as well. If those future I-patterns are not STBM consumers, canon needs a separate manager contract for their multi-tranche lifecycle.  
Recommended action: Either extend STBM applicability to all triple-barrier patterns or define a distinct short-horizon/event-overlay execution manager before drafting I-pattern execution specs.

[LOW] Regression-Checklist.md:567 — Version notes still say M1 is the only fully built pattern  
The checklist now contains M3 and M4 pattern-specific invariants, but the version note says M1 is the only fully-built pattern. This is stale metadata, not a behavioral issue.  
Recommended action: Update the version note to reflect M1/M3/M4 built status.

[NOTE] SyntheticTripleBarrierManager.md:379 — framework_exit_cleanup marks the STBM group closed before the framework exit market order  
The cleanup contract cancels GTC orders, transitions the STBM state to CLOSED, and then the consuming execution layer submits the framework-level market-order exit. This can be valid if CLOSED means "STBM order group closed," not "broker position flat," but the wording currently says position state.  
Recommended action: Clarify that framework_exit_cleanup closes the managed GTC lifecycle, while the pattern execution layer remains responsible for tracking the subsequent flatten order until broker fill.

## Closing Observations

The root canon is close enough to be auditable, but not yet clean enough to be treated as an agent-readable implementation source of truth without guardrails. The largest risk is contradictory authority: newer Wave 2.11/STBM/checklist rules are usually better specified, but older Architecture and track-type language remains live and would produce different code. The next maintenance pass should prioritize deleting or explicitly superseding stale canonical text, not adding more explanatory prose around it.
