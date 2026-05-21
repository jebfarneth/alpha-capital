# M4 Fresh-Eyes Audit 4

**Timestamp:** 2026-05-14 07:57:05 EDT  
**Scope:** M4 SPEC, EXPOSURE, EXECUTION, VALIDATION, DATA; root references as needed  
**Posture:** Antagonistic, fresh-session, read-only

## Summary by Severity

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 1 |
| NOTE | 1 |

## Findings

[HIGH] DATA.md:210 — expected_round_trip_cost units conflict with Architecture canonical units  
M4 DATA says `expected_round_trip_cost` is stored "in basis points." Architecture.md:1173 defines the same field in the same decimal-return units as `combined_expected_edge`. If implementation follows M4 literally, `net_expected_edge = combined_expected_edge - expected_round_trip_cost` can be off by 10,000x or require hidden conversion logic.  
Recommended action: Change M4 DATA field semantics to decimal-return units, e.g. `0.012` for 120 bps, and state that display/reporting may convert to bps.

[HIGH] DATA.md:302 — floor_stop can close the position but cannot be represented in tranche fills  
M4 DATA declares `final_exit_reason = "floor_stop"` for a post-T2 T1-floor stop close, but `m4_tranche_fills.tranche_label` omits `floor_stop`. A valid M4 lifecycle therefore has no legal child-table label for the actual broker fill that closes the remaining 40% at the T1 floor.  
Recommended action: Add `floor_stop` to `m4_tranche_fills.tranche_label` and specify `fill_type = "stop_loss"` or a more precise protective-stop value.

[HIGH] EXECUTION.md:235 — simultaneous T1/stop handling relies too heavily on broker overfill rejection  
The race row says Alpaca rejects the over-fill portion automatically if a 100% stop and 30% T1 fill would over-exit. With independent GTC sells, implementation should not rely on broker rejection as the safety contract, especially in a long-only system where margin settings or order timing could create an unintended short or rejected-state ambiguity.  
Recommended action: Reword to match STBM's safer contract: accept broker-confirmed fills, immediately reconcile broker position vs internal filled quantity, cancel remaining orders, flatten residual or cover unintended short if needed, and escalate if reconciliation is not provably flat.

[MEDIUM] EXECUTION.md:151 — time barrier is classified as pattern-native while DATA classifies it as framework-level  
EXECUTION lists "six pattern-native exit conditions" and includes the time barrier, while DATA.md:328-335 classifies `time_barrier` as one of the four framework-level exit reasons. The routing behavior is mostly correct because EXECUTION invokes `framework_exit_cleanup`, but the taxonomy is inconsistent and can mislead future enum or cleanup edits.  
Recommended action: Define time barrier as the pattern-specified vertical barrier that is represented by the framework-level `time_barrier` exit reason, then normalize wording in SPEC and EXECUTION.

[MEDIUM] EXECUTION.md:231 — M4 race table omits STBM canonical out-of-order tranche cases  
STBM requires handling T2-before-T1 and T3-before-T2 cases. M4 covers T1/T2 close succession but does not explicitly cover T3-before-T2, and it does not state when `tranche_state_anomaly` is emitted despite declaring the enum in DATA.md.  
Recommended action: Add explicit rows for T2-before-T1, T3-before-T2, and unexpected tranche-state corruption; emit `tranche_state_anomaly` only for genuinely unexpected state, not deterministic out-of-order fills.

[MEDIUM] DATA.md:317 — framework_exit_cleanup is used as a tranche_label even though cleanup is not the fill  
`framework_exit_cleanup` cancels active GTC orders before a framework-level market exit; it is not itself a broker fill. Using it as a child-table `tranche_label` can blur the difference between cleanup completion and the actual market-order fill caused by `time_barrier`, `universe_ejection`, `circuit_breaker_flatten`, or `optimizer_rebalance`.  
Recommended action: Replace the child-table label with actual framework exit labels or add a separate cleanup event table/field while keeping the fill row tied to the real market-order reason.

[MEDIUM] EXECUTION.md:36 — Class A cancel-replace lifetime lacks a hard maximum entry-attempt boundary  
The entry order persists across sessions until fill, signal decay, veto, circuit breaker, or universe ejection. For a 52-week-high breakout, "signal remains alive" is operationally fuzzy after the original breakout day, especially if price remains near but not above the prior high.  
Recommended action: Add a max entry-attempt window, such as same-session only or next-session retry only, and define the exact nightly condition that keeps an unfilled M4 entry alive.

[LOW] SPEC.md:174 — stale cross-reference still says exit_reason enum  
DATA.md now uses `final_exit_reason`, but SPEC still references "exit_reason enum." VALIDATION cross-references have the same stale wording. This is cosmetic but worth cleaning to prevent future search confusion.  
Recommended action: Replace stale `exit_reason` references with `final_exit_reason` where referring to M4 DATA.

[NOTE] VALIDATION.md:264 — January diagnostic is correctly deferred, but the promotion horizon is effectively non-actionable  
The January seasonality test is deferred until about 24 January observations, estimated at roughly 12 years. That is statistically honest at M4's expected fill rate, but it means the diagnostic is not practically useful for V1 or near-term governance.  
Recommended action: Leave as deferred, or explicitly label it as "research archival / not expected to inform V1-V3 operations."

## Closing Observations

M4 is much cleaner than earlier audit passes. The major remaining issues are not thesis-level; they are implementation-contract issues in DATA/EXECUTION. Fix the cost-unit semantics, the missing `floor_stop` tranche label, and the overfill-race language before handing M4 to an agentic implementation system.
