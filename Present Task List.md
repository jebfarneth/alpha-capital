# Present Task List
_Last updated: 2026-06-17_

This is the current working order. Older A1/A2/A3/I11-at-open build prompts are
superseded unless explicitly revived.

## TL;DR

I12 is still the lead research strategy. The live read-only fill-test machine is
built, and Alpaca SIP quotes work. The current frozen model is mechanically
loadable, but explicitly non-promotable because the I12 corpus is deferred-PIT.
The real blocker is a PIT-clean as-of-entry I12 rebuild, not another attempt to
promote the old full-day-volume corpus.

## Current Truth

- Stage-0 read-only I12 fill-test exists: detect, score, rank, quote, log, exit
  quote, Gate-0 report.
- Stage-0 places no orders.
- Alpaca SIP quote probe works after the account upgrade.
- Context artifact exists on the VM for 2026-06-17:
  `artifacts/stage0/i12_context_2026-06-17.json`.
- Current research-shadow model:
  `stage1_i12_403a5ae359cd_accecdda`.
- It is loadable by Stage-0 preflight, but non-promotable with
  `deferred_pit_model`.
- The next useful engineering output is artifact-SHA/training-params preflight
  hardening. The next useful strategy output is a PIT-clean I12 corpus and
  replacement model.

## Track A - Finish Current Model-Release Hardening

This is a release-safety pass, not a production promotion.

1. Keep older I12 model rows rejected.
2. Keep `stage1_i12_403a5ae359cd_accecdda` as `shadow`, not production.
3. Keep PIT provenance fail-closed: `pit_deferred` must be explicit false and
   `pit_failed_row_count` must be exactly zero for promotion.
4. Add Stage-0 preflight validation for the committed artifact bytes and
   artifact-level `training_params`, not only registry metadata.
5. Verify Stage-0 preflight loads by explicit `--model-id` and smoke-scores in
   scratch.

Done means the report includes:

- model id
- registry schema
- artifact path
- artifact SHA
- manifest SHA
- feature schema hash
- exact feature list
- null rate for every training feature
- `pit_failed_row_count`
- strict JSON verification
- OOS lift comparison against the prior research result
- Stage-0 preflight result
- explicit statement: research-shadow only, non-promotable with
  `deferred_pit_model`

## Track B - PIT-Clean I12 Rebuild

This is the load-bearing work before any real promotion.

1. Finish operational integrity first: source attempts, quote replays, and cost
   replays must all be resumable/idempotent in the same scratch schema.
2. Generate historical candidate rows at fixed early intraday decision times
   such as 9:35, 9:40, 9:45, and 10:00.
3. Use only fields knowable at that timestamp:
   prior closed bars, opening gap, early cumulative volume, projected volume
   pace, early price action, live-style spread/size proxies, and PIT-safe
   catalyst/hazard flags.
4. Ban full-day volume, close, high/low after decision time, future bars, and
   any feature filled after the decision timestamp.
5. Replay scoped historical SIP quotes only around candidate event windows:
   decision/entry, same-day near-close or 15:55, and next-session open.
6. Recompute tradeability and returns with realistic quote assumptions:
   entry at ask, exit at bid, stale/missing/wide/thin quotes skipped to cash.
7. Evaluate the simple PIT-clean detector first, with no neural net and no extra
   model complexity:
   - all-candidate edge
   - same-day exit edge
   - next-open exit edge
   - spread/depth coverage
   - skip rate as cash
   - cost sensitivity
8. Retrain only if the PIT-clean corpus preserves meaningful lift and survives
   real spread/liquidity costs.

If the simple PIT-clean live-visible set is weak, do not declare I12 dead and
do not jump straight to a neural net. Run the diagnostic ladder:

1. **Signature study:** within the live-visible candidate set, compare eventual
   winners versus losers at each decision time. Do not study only old full-day
   winners; that recreates deferred-PIT selection bias.
2. **Engineered early-curve features:** add explicit curve features to the
   existing GBRT first:
   - cumulative volume at 9:30/9:35/9:40/9:45/10:00
   - volume ramp slope and acceleration
   - spike-then-sustain versus spike-then-fade
   - early return path and stabilization/reversal
   - spread/depth behavior
   - PIT-safe catalyst/hazard flags
3. **Sequence model only if earned:** consider a raw-tape NN only if the
   engineered-feature GBRT leaves clear out-of-time signal on the table.

The discriminant must separate winners from losers among names a live detector
would actually surface. Describing old winners is not enough.

## Track C - Run Stage-0 Read-Only Instrumentation

This can run with the research-shadow model for plumbing and quote coverage, but
it is not a promotion test until a PIT-clean model exists.

Probe command:

```bash
cd /home/jebwfarneth/alpha-capital/engine
set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_live_fill_test \
  --alpaca-probe-symbol AAPL \
  --feed sip
```

One-shot Stage-0 run with the research-shadow model:

```bash
cd /home/jebwfarneth/alpha-capital/engine
set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_live_fill_test \
  --schema scratch_i12_stage0_YYYYMMDD \
  --create-tables \
  --model-registry-schema i12_rebuild_20260615_codex \
  --context-artifact artifacts/stage0/i12_context_YYYY-MM-DD.json \
  --trading-date YYYY-MM-DD \
  --model-id stage1_i12_403a5ae359cd_accecdda \
  --feed sip \
  --top-k 10 \
  --intended-order-usd 250 \
  --max-spread-bps 200 \
  --once
```

Monitor run:

- same command
- remove `--once`
- run in VM terminal or tmux so local laptop crashes do not matter

Gate-0 promotion report:

```bash
.venv/bin/python -m alpha.jobs.run_i12_live_fill_test \
  --schema scratch_i12_stage0_YYYYMMDD \
  --model-id stage1_i12_403a5ae359cd_accecdda \
  --feed sip \
  --gate0-report \
  --fail-on-gate0-fail
```

Gate 0 passes only if the logged intended names are genuinely tradeable at the
planned ~$250 size. Skips count as cash. A high historical edge does not matter
if live names are too wide, too stale, halted, or too thin.

With the current research-shadow model, Gate 0 should remain non-promotable due
`deferred_pit_model`.

## Track D - After PIT-Clean Gate 0

Only after read-only logs are clean:

1. Write the Stage-1 paper-execution prompt.
2. Run paper orders, still no live capital.
3. Compare modeled fills vs actual paper fills.
4. Decide whether a small real-money pilot is justified.

## Track E - Uncorrelated Sleeves

This is strategy work, not the current blocker.

Goal: reduce I12 tail risk and drawdown, not find more raw return.

Current best candidates:

- market-neutral cross-sectional residual / pairs sleeve
- slow equity sleeve that is not just another I12-shaped small-cap rebound
- options/vol sleeve only if data and execution are realistic enough to test

Do not build these ahead of PIT-clean I12 validation unless the I12 rebuild
fails and the strategy focus changes.

## Do Not Do Yet

- Do not promote the current research-shadow model as production-clean.
- Do not place paper orders from Stage-0.
- Do not place live orders.
- Do not lever.
- Do not add more I12 names to solve drawdown; correlation, not count, is the
  problem.
- Do not call any I12 model production-clean unless the corpus is PIT-clean and
  the registry says `pit_deferred=false`, `pit_failed_row_count=0`.
