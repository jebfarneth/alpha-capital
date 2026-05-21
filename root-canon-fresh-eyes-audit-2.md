# Root Canonical Documents Fresh-Eyes Audit 2

**Timestamp:** 2026-05-14 08:01:06 EDT  
**Scope:** Architecture.md, Patterns.md, Universe.md, Validation.md, Regression-Checklist.md, SyntheticTripleBarrierManager.md  
**Posture:** Antagonistic, fresh-session, read-only

## Summary by Severity

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| HIGH | 7 |
| MEDIUM | 5 |
| LOW | 1 |
| NOTE | 1 |

## Findings

[HIGH] Architecture.md:985 — I-track framework-exit table still contradicts corrected I-track definition  
Architecture.md:930 correctly defines I-track as intraday-to-5-day and says multi-session I-track uses all four framework exits. The Framework-Level Exit Reasons table still says `time_barrier` and `optimizer_rebalance` are not applicable to I-track because positions are intrinsic single-session holds.  
Recommended action: Update the table and scoping clarification to match line 930: same-session I-track gets only same-session applicable exits, while multi-session I-track gets `time_barrier`, `universe_ejection`, `circuit_breaker_flatten`, and `optimizer_rebalance`.

[HIGH] Regression-Checklist.md:552 — STBM consumer scope remains M-track-only despite root Architecture extending STBM to all triple-barrier patterns  
Architecture.md:1243 says STBM owns lifecycle management for all triple-barrier patterns regardless of track. Regression-Checklist.md still tracks STBM consumers as "All M-track patterns using synthetic triple-barrier (M3, M4, M5, M6 once drafted)," which would skip I-track triple-barrier consumers during infrastructure-change audits.  
Recommended action: Change the checklist consumer scope to "all triple-barrier patterns with partial exits, M-track and I-track," and list future I-pattern consumers once drafted.

[HIGH] SyntheticTripleBarrierManager.md:410 — framework_exit_cleanup consuming-pattern table omits future I-track triple-barrier consumers  
STBM scope says it applies to any pattern requiring triple-barrier exits with partial take-profits, but the cleanup consumer table lists only M3, M4, future M5, and future M6. Because Patterns.md assigns triple-barrier exits to I1/I3/I5/I7/I8/I9/I10 and overlays I2/I4/I6, future implementation agents could incorrectly withhold cleanup timestamps and framework-exit cleanup from I-track patterns.  
Recommended action: Either generalize the consumer table to "all STBM-consuming triple-barrier patterns" or add an explicit future-I-pattern row template with `i*_validation_metadata.framework_exit_cleanup_timestamp`.

[HIGH] Architecture.md:824 — leverage deployment formula still conflicts with stated effective-deployment targets  
The canonical formula `target_gross_exposure = active_leverage_tier * deployable_fraction` with a 75% deployable fraction implies 112.5% exposure at 1.5x and 150% at 2.0x. Architecture.md:1534 instead says effective deployment rises to 90% and 120%. This is a direct sizing/risk-control ambiguity.  
Recommended action: Define a single table or formula for levered target exposure, including whether deployable fraction changes by leverage tier, and make all examples derive from it.

[HIGH] Architecture.md:882 — validation_weight_multiplier exists in Validation.md but is missing from Architecture's optimizer-input formula  
Validation.md:222-236 introduces `validation_weight_multiplier` as the governance-safe way to cap optimizer contribution without mutating `pattern_weights.current_weight`. Architecture's Stage 2 formula does not include it, so an implementation agent following Architecture alone would ignore validation gating.  
Recommended action: Add `validation_weight_multiplier_i` to the optimizer-input expected-edge formula and describe its position relative to pattern_weight, shrinkage, cost, and missed-fill adjustment.

[HIGH] Architecture.md:679 — PDT statement is premature and overbroad as of 2026-05-14  
Architecture says PDT restrictions have been removed and intraday trading is unconstrained at any account size. Current regulatory material says the new intraday margin requirements are effective June 4, 2026, with broker transition permitted until October 20, 2027; FINRA's current investor page still describes the $25,000 PDT requirement under existing rules. This can materially affect V1 trading assumptions below $25k.  
Recommended action: Replace with date-specific language: PDT relief has been approved but is not fully operational for all brokers until implementation; verify Alpaca account eligibility before assuming unconstrained intraday flow. Sources: [Investor.gov PDT glossary](https://www.investor.gov/introduction-investing/investing-basics/glossary/pattern-day-trader), [FINRA day trading](https://www.finra.org/investors/investing/investment-products/stocks/day-trading), [Schwab summary of SEC approval](https://www.schwab.com/learn/story/sec-approves-scrapping-25000-day-trader-minimum).

[HIGH] Built pattern DATA files — expected_round_trip_cost unit semantics drift from Architecture decimal-return canon  
Architecture.md:1173 defines `expected_round_trip_cost` in the same decimal-return units as `combined_expected_edge`. M1 DATA.md:256, M3 DATA.md:294, and M4 DATA.md:210 describe the field as basis points. The schemas now include the column, but a units mismatch can corrupt `net_expected_edge` arithmetic by a factor of 10,000 unless hidden conversion logic is added.  
Recommended action: Normalize all built pattern DATA field-semantics rows to decimal-return units, with bps only as display notation.

[MEDIUM] Universe.md:86 — launch gate liquidity thresholds do not match Architecture liquidity-score thresholds  
Universe.md's binding launch gate uses "Median 30-day ADV >= $250K" and 50 bps spread. Architecture.md:320-324 requires 20-day median dollar volume >= $1M, 60-day median dollar volume >= $750K, and 20-day effective spread <= 2.5%. The launch gate says it validates canonical liquidity requirements but uses different horizons and thresholds.  
Recommended action: Rewrite the launch gate to apply the Architecture liquidity score directly, then optionally add stricter supplemental diagnostics if desired.

[MEDIUM] Architecture.md:681 — multi-day track horizon says 5-30 days while root taxonomy says 5-20 days  
Architecture.md:13 defines multi-day as 5-20 day holds. The trade-flow section says M1-M7 are held 5-30 days depending on time barrier. Current Patterns.md rows top out at 20 days for M2, so 30 days appears stale.  
Recommended action: Normalize to 5-20 days unless a specific future pattern is intended to exceed 20 days.

[MEDIUM] Architecture.md:1247 — stop-ratcheting text is scoped to M-track despite Patterns.md inheritance applying to all triple-barrier patterns  
The exit-orders section says "For M-track triple-barrier patterns, the hard stop is NOT static," but Patterns.md:151 says all triple-barrier patterns inherit multi-stage stop ratcheting unless explicitly overridden. That wording can lead future I-patterns to omit state-aware stop fields.  
Recommended action: Change to "For all triple-barrier patterns inheriting Patterns.md multi-stage rules..." and only carve out explicit overrides.

[MEDIUM] SyntheticTripleBarrierManager.md:137 — tranche quantity contract is M3-specific in the generic order-placement section  
The generic STBM Order Placement Contract immediately lists "For M3, tranche quantities are..." and does not give the general rule that tranche percentages are supplied per pattern. M4 is later listed, but the top-level contract still reads as if 30/30/40 is M3-owned rather than continuous-factor default / pattern-configurable.  
Recommended action: Move tranche percentages into a generic `tranche_config` input, then list M3/M4 examples in the pattern-specific parameter section.

[MEDIUM] Architecture.md:414 — I4 remains classified as FMP-only despite halt-specific source requirements being undefined  
The revised FULL dependency section correctly adds FDA.gov and SEC EDGAR for I2/I6, but I4 Halt and Resume is still in the "FMP Ultimate baseline only" bucket. The root docs do not identify an authoritative trading-halt source or prove FMP WebSocket can distinguish news-pending halts from LULD halts as required by Patterns.md.  
Recommended action: Add an I4 structured halt-status dependency, or explicitly document that FMP provides authoritative halt/resume classification including news-halt vs LULD filtering.

[LOW] Regression-Checklist.md:83 — checklist still says STBM lifecycle for multi-tranche M-track patterns only  
Layer 1 Wave 2.11 says STBM owns lifecycle for multi-tranche M-track patterns, while newer Architecture language extends it to all triple-barrier patterns. This is redundant with the higher-severity consumer-scope issue but should still be cleaned.  
Recommended action: Replace "M-track" with "all STBM-consuming triple-barrier patterns."

[NOTE] Architecture.md:1562 — return projection consolidation is now correctly handled  
The prior stale return projection issue is substantially fixed: the lower 11-13% / $4k-$6k shorthand is now explicitly marked superseded and points back to the canonical trade-flow projection block. No further action beyond keeping future projections centralized.

## Closing Observations

This pass shows real convergence: several earlier root-canon problems were fixed cleanly. The remaining highest-risk issues are now more surgical: I-track exit applicability, STBM consumer scope, leverage exposure math, validation-gate integration into the optimizer formula, PDT timing, and cost-unit semantics in built pattern DATA files. Fix those before using the root canon as an implementation source for I-track patterns or leverage-aware sizing.
