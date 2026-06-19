# Power-Outage Checkpoint - 2026-06-18

Local recovery note. Contains no secrets. Keep it current while Stage-0, the
I12 model freeze, and the PIT-clean rebuild are in motion.

## State In One Line

I12 PIT-clean rebuild/replay exists and was pushed through `origin/main`
`a2b45bc`; the VM died during the 25-day strict 09:40 run after finishing
`2026-05-15`. Local workspace now has additional uncommitted PIT sparse-mode,
source-denominator, and progress-integrity hardening. Do not resume real
research runs until deciding whether to commit/push/pull those local hardening
changes or intentionally resume from `a2b45bc`.

## Git / Workspace

- Local branch: `main`.
- Remote visible during this checkpoint: `origin/main` at
  `a2b45bc Add PIT-clean I12 rebuild replay`.
- Local tracked changes currently include uncommitted PIT-clean rebuild
  hardening:
  - `engine/alpha/jobs/i12_pit_rebuild.py`
  - `engine/alpha/jobs/run_i12_pit_rebuild.py`
  - `engine/alpha/db/models.py`
  - `engine/migrations/versions/fd0123456791_add_i12_pit_rebuild_tables.py`
  - `engine/tests/test_i12_pit_rebuild.py`
- Known untracked local file: `engine/.cache/`.
- Do **not** represent the research-shadow I12 model as production-clean. It is
  read-only/research-shadow until a PIT-clean corpus replaces it.

Latest local validation for the uncommitted hardening:

- Targeted PIT progress regressions: `3 passed`
- PIT rebuild + adapters: `444 passed`
- Broader focused suite: `425 passed, 2 skipped`
- `git diff --check`: passed
- `py_compile`: passed

## VM Failure / PIT Run Recovery

The VM failed during this run:

```bash
cd /home/jebwfarneth/alpha-capital/engine

set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_20260618 \
  --create-tables \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --feed sip \
  --intended-order-usd 250 \
  --max-spread-bps 200 \
  --max-quote-age-seconds 60 \
  --progress-artifact artifacts/stage0/i12_pit_m1_0940_20260618_progress.json
```

Last known progress before the VM failure:

```json
{
  "last_trading_date": "2026-05-15",
  "hur_rows_loaded": 29104,
  "candidate_row_count": 29104,
  "candidate_passed": 147,
  "candidate_failed": 28957,
  "quote_replayed_candidates": 147,
  "quote_replay_row_count": 441,
  "cost_replay_row_count": 294,
  "daily_fetch_error_count": 0,
  "minute_fetch_error_count": 0
}
```

Interpretation:

- Run had completed through `2026-05-15`.
- The requested window was `2026-05-01` through `2026-06-05`.
- `quote_replay_row_count = 147 * 3`, correct.
- `cost_replay_row_count = 147 * 2`, correct.
- No provider fetch errors were recorded before failure.
- Most strict-mode exclusions were `partial_minute_path`, so sparse-mode
  comparison remains important.

After VM restart, first confirm whether the scratch DB rows survived:

```bash
cd /home/jebwfarneth/alpha-capital/engine

set -a
source .env
set +a

.venv/bin/python - <<'PY'
import os
from sqlalchemy import create_engine, text

schema = "scratch_i12_pit_m1_0940_20260618"
engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)

with engine.connect() as conn:
    rows = conn.execute(text(f"""
        select decision_date, count(*) rows,
               sum(case when candidate_status='passed' then 1 else 0 end) passed,
               max(updated_at) last_update
        from {schema}.i12_pit_candidates
        where is_active
        group by decision_date
        order by decision_date desc
        limit 12
    """)).mappings().all()
    for r in rows:
        print(dict(r))

    print("totals")
    for table in ["i12_pit_candidates", "i12_pit_quote_replays", "i12_pit_cost_replays"]:
        n = conn.execute(text(f"select count(*) from {schema}.{table}")).scalar()
        print(table, n)
PY
```

If the rows survived and we intentionally continue from the VM's current code,
rerun the same command above. The PIT rebuild is date-boundary committed and
idempotent, so it should reuse existing active candidates/quotes/cost rows.

Preferred recovery before more serious research runs:

1. Commit/push the local PIT hardening if still clean.
2. Pull it on the VM.
3. Use a fresh scratch schema for defensible final evidence, or explicitly
   accept that `scratch_i12_pit_m1_0940_20260618` was created before the latest
   local hardening.
4. Rerun strict 09:40.
5. Then run sparse 09:40 apples-to-apples.

Fresh strict schema command template:

```bash
cd /home/jebwfarneth/alpha-capital/engine

set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_strict_YYYYMMDD \
  --create-tables \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --feed sip \
  --intended-order-usd 250 \
  --max-spread-bps 200 \
  --max-quote-age-seconds 60 \
  --progress-artifact artifacts/stage0/i12_pit_m1_0940_strict_YYYYMMDD_progress.json
```

Fresh sparse schema command template:

```bash
cd /home/jebwfarneth/alpha-capital/engine

set -a
source .env
set +a

.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_sparse_YYYYMMDD \
  --create-tables \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --minute-path-mode sparse_zero_fill \
  --feed sip \
  --intended-order-usd 250 \
  --max-spread-bps 200 \
  --max-quote-age-seconds 60 \
  --progress-artifact artifacts/stage0/i12_pit_m1_0940_sparse_YYYYMMDD_progress.json
```

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
commit: c84cfac Freeze I12 stage1 model
registry: i12_rebuild_20260615_codex.ml_model_registry
artifact: engine/artifacts/ml/stage1_i12/stage1_i12_403a5ae359cd_accecdda.pkl
artifact_sha256: c8d104a664a27455c9e0e1b3677ea0c97f0030cca23e7d833a92538216362416
manifest: engine/alpha/ml/manifests/stage1_i12_manifest_v1.json
manifest_version: stage1_i12_research_shadow_v2
manifest_sha256: 6032ff99ce5b8d12c24f6b4b7967e170cf07d710c6b695e7e5b0fcecd46ed0e4
feature_schema_hash: 403a5ae359cddc5a4927db8bf874addb209b17fc5d82415f134357adef748ee1
status: shadow
promotability: non-promotable, deferred_pit_model
```

Facts:

- Artifact/manifest/schema checks pass.
- Stage-0 artifact preflight pins the artifact SHA and strict horizon metadata.
- Pickle loaded and predicted normally.
- OOS lift matches the research target, roughly 2x top-decile lift.
- Older I12 model rows are rejected in the registry.
- The current row carries `pit_deferred=true` / failed PIT provenance and is blocked from promotion.

Interpretation:

The model is mechanically valid and useful as a research-shadow reference. It is
not a clean production model because the historical I12 corpus selected fires
using full-day volume, which is future information at the live decision minute.

## Current Model Hardening Status

The model-release hardening item from the prior checkpoint was completed:
Stage-0 artifact preflight now pins the expected artifact SHA and verifies
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
  PIT-clean rolling candidate-tape rebuild using as-of-timestamp intraday
  features.
- The I12 target is a rolling top-K sleeve, not a single 09:40 decision. Names
  can qualify at different timestamps through the day, with one intended trade
  per ticker/day and skips counted as cash.
- The broader portfolio target is not "more I12 names"; it is a K around 20
  across independent sleeves.
- I12 research suggests the return sleeve; uncorrelated sleeves are for
  drawdown/tail control.
- No leverage until the tail is controlled.
- Live capital comes after PIT-clean validation, Stage-0 read-only logs, and then
  Stage-1 paper execution.

## Resume Order

1. Decide whether to commit/push the current local PIT hardening. It has passed
   focused validation and fixes sparse-mode/report/progress integrity.
2. If pushing, pull on the VM before resuming research runs.
3. After VM recovery, verify whether `scratch_i12_pit_m1_0940_20260618`
   survived. If using it, remember it was started before the latest local
   hardening.
4. Prefer a fresh scratch schema for final evidence.
5. Run strict 09:40 for `2026-05-01` through `2026-06-05` as one timestamp
   slice, not as the full live strategy.
6. Run sparse-zero-fill 09:40 over the same dates.
7. Compare strict versus sparse reports. Sparse exists to test whether thin names
   with no minute trades were being excluded unfairly.
8. Extend the same PIT-clean replay to multiple decision times and evaluate a
   rolling top-K tape: first qualifying timestamp, no duplicate ticker/day buys,
   later qualifiers filling empty slots, and explicit replacement rules if any.
9. Evaluate the simple PIT-clean detector first: all candidates, false
   positives included, skips-as-cash, same-day exit, next-open exit, real
   spread/depth costs.
10. Retrain/register a production-clean replacement model only if PIT-clean
   metrics hold.
11. If simple PIT-clean metrics are weak, run the early-signature diagnostic:
    winners versus losers within the live-visible candidate tape, including
    names that first qualify later in the day, then engineered early-curve
    features in the existing GBRT. Consider a raw-tape neural net only if that
    auditable baseline leaves clear out-of-time signal on the table.
12. Then run Stage-0 promotion and write the paper-execution Stage-1 prompt.
