# Power-Outage Checkpoint - 2026-06-17

Local recovery note. Contains no secrets. Keep it current while Stage-0, the
I12 model freeze, and the PIT-clean rebuild are in motion.

## State In One Line

I12 Stage-0 read-only fill-test code is built and pushed through `origin/main`
`4e5c658`; Alpaca SIP quotes work. Local `HEAD` contains a loadable
research-shadow I12 model, but it is explicitly non-promotable because the
corpus is deferred-PIT. Do not treat this as a production model.

## Git / Workspace

- Local branch: `main`.
- Remote visible during audit: `origin/main` at `4e5c658 Harden I12 Stage-0 fill test gates`.
- Local `HEAD` during this checkpoint: `cdf0287 Freeze I12 stage1 model`.
- Status during audit: local `main` is ahead of `origin/main` by 1 because of the model-freeze commit.
- Do **not** represent the local model-freeze commit as production-clean. It is read-only/research-shadow until a PIT-clean corpus replaces it.
- Known unrelated local files: `README.md`, `HOW_IT_WORKS.md`, `POWER_OUTAGE_CHECKPOINT.md`, `Present Task List.md`, `engine/.cache/`.

## Stage-0 Fill-Test Status

Stage-0 is read-only:

- no orders
- no paper orders
- no live capital
- quote/data reads only
- scratch schema only for fill logs and copied model row

Built pieces:

- `alpha.jobs.run_i12_live_fill_test`
- scratch-bound `i12_fill_log`
- Alpaca read-only quote/snapshot methods
- model-registry source-schema copy into scratch
- strict Gate-0 report denominators and tradeability checks
- explicit quote-size basis: `shares_post_2025_11_03`

Alpaca SIP entitlement:

```bash
cd /home/jebwfarneth/alpha-capital/engine
set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_live_fill_test \
  --alpaca-probe-symbol AAPL \
  --feed sip
```

Expected shape:

```json
{"ok": true, "read_only": true, "symbol": "AAPL", "bid": ..., "ask": ..., "timestamp": "..."}
```

VM `.env` fix already applied:

```text
SEC_USER_AGENT="Alpha Capital Jeb Farneth jebwfarneth@outlook.com"
```

## Context Artifact

Built on VM:

```text
artifacts/stage0/i12_context_2026-06-17.json
```

Observed shape:

- JSON object with `artifact_version`, `context_date`, `contexts`, `ticker_count`
- `ticker_count`: 13,700

Rebuild command if needed:

```bash
cd /home/jebwfarneth/alpha-capital/engine
set -a
source .env
set +a

mkdir -p artifacts/stage0

.venv/bin/python -m alpha.jobs.run_paper_execution \
  --context-artifact artifacts/stage0/i12_context_YYYY-MM-DD.json \
  --trading-date YYYY-MM-DD \
  --build-context \
  --pattern-id I12
```

## Current Frozen Model

This model is loadable and useful for read-only Stage-0 plumbing, but not
production-clean:

```text
model_id: stage1_i12_403a5ae359cd_accecdda
commit: cdf0287 Freeze I12 stage1 model
registry: i12_rebuild_20260615_codex.ml_model_registry
artifact: engine/artifacts/ml/stage1_i12/stage1_i12_403a5ae359cd_accecdda.pkl
manifest: engine/alpha/ml/manifests/stage1_i12_manifest_v1.json
manifest_version: stage1_i12_research_shadow_v2
manifest_sha256: 6032ff99ce5b8d12c24f6b4b7967e170cf07d710c6b695e7e5b0fcecd46ed0e4
feature_schema_hash: 403a5ae359cddc5a4927db8bf874addb209b17fc5d82415f134357adef748ee1
status: shadow
promotability: non-promotable, deferred_pit_model
```

Facts:

- Artifact/manifest/schema checks pass.
- Pickle loaded and predicted normally.
- OOS lift matches the research target, roughly 2x top-decile lift.
- Older I12 model rows are rejected in the registry.
- The current row carries `pit_deferred=true` / failed PIT provenance and is blocked from promotion.

Interpretation:

The model is mechanically valid and useful as a research-shadow reference. It is
not a clean production model because the historical I12 corpus selected fires
using full-day volume, which is future information at the live decision minute.

## Remaining Model Hardening

The last hard audit found one remaining model-release hardening item before a
push: Stage-0 artifact preflight should pin the expected artifact SHA and verify
artifact-level `training_params` (`horizon_sessions == 1`, `signal_horizon ==
"1d"`), not only registry metadata.

Strategic fix still required: build the PIT-clean I12 corpus and replacement
model. The current model should remain research-shadow/non-promotable.

## Stage-0 Run Command

This is read-only instrumentation. It should not be interpreted as a model
promotion test while using the deferred-PIT research-shadow model.

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

Then remove `--once` for the market-hours monitor.

## Gate-0 Promotion Bar

Gate 0 is about tradeability, not profitability:

- at least 20 intended names
- at least 3 intended trading days
- SIP feed only
- explicit model id only
- quote coverage strong
- exit quote coverage strong when due
- no mixed policy hash
- no mixed context artifact hash for the same day
- no missing or unsupported quote-size basis
- tradeable rate at least 70%
- skipped names counted as cash, not dropped
- no deferred-PIT model provenance

## Strategy State

- I12 remains the lead research sleeve, but the old detector is not production-clean.
- Full-day volume cannot be a live detector gate. The next real gate is a
  PIT-clean as-of-entry rebuild using early intraday features.
- The target is not "more I12 names"; it is a K around 20 across independent sleeves.
- I12 research suggests the return sleeve; uncorrelated sleeves are for
  drawdown/tail control.
- No leverage until the tail is controlled.
- Live capital comes after PIT-clean validation, Stage-0 read-only logs, and then
  Stage-1 paper execution.

## Resume Order

1. Finish PIT-clean rebuild operational integrity: source, quote, and cost
   replay rows must all recover cleanly from transient provider errors without
   poisoning the scratch schema.
2. Push only if docs and non-promotable provenance are accurate.
3. Pull on VM.
4. Run Stage-0 read-only instrumentation with the research-shadow model if useful.
5. Build PIT-clean I12 as-of-entry candidate corpus with dual exits.
6. Run scoped Alpaca SIP quote replay for candidate event windows only.
7. Evaluate the simple PIT-clean detector first: all candidates, false
   positives included, skips-as-cash, same-day exit, next-open exit, real
   spread/depth costs.
8. Retrain/register a production-clean replacement model only if PIT-clean
   metrics hold.
9. If simple PIT-clean metrics are weak, run the early-signature diagnostic:
   winners versus losers within the live-visible candidate set, then engineered
   early-curve features in the existing GBRT. Consider a raw-tape neural net only
   if that auditable baseline leaves clear out-of-time signal on the table.
10. Then run Stage-0 promotion and write the paper-execution Stage-1 prompt.
