# I12 PIT 09:40 Report Commands

Run from the VM repo checkout:

```bash
cd /home/jebwfarneth/alpha-capital/engine
test -f .env
```

The shard launchers below load `.env` in strict shell mode and fail before
starting Python if the file is missing or cannot be sourced.

## Shard Launchers

The strict and sparse launchers run a single schema/table preflight in the
control shell before fan-out. Shard workers then run without `--create-tables`,
so a fresh parallel launch does not race on scratch table creation.

Each worker also runs the PIT runner's no-progress watchdog:

```bash
MAX_NO_PROGRESS_MINUTES=20
```

The runner default is 20 minutes for normal rebuild runs, and these launchers
also pass the value explicitly to every shard as `--max-no-progress-minutes`.
A provider/socket wedge exits nonzero instead of polling forever. Override it
only when you have a reason to tolerate longer provider stalls:

```bash
MAX_NO_PROGRESS_MINUTES=30 scripts/run_i12_pit_0940_strict_shards.sh
```

Existing tmux panes do not inherit new defaults or launcher matching rules. Kill
or relaunch old panes after pulling this code if they were started before the
no-progress monitor was added.

Strict:

```bash
scripts/run_i12_pit_0940_strict_shards.sh
```

Sparse zero-fill:

```bash
scripts/run_i12_pit_0940_sparse_shards.sh
```

Both scripts reuse their tmux session if it already exists. They only skip an
existing shard window when that pane is actively running the expected Python
command for the same schema, date range, path mode, progress artifact, and
exact `--max-no-progress-minutes` value. Old panes launched without the monitor,
or with a different monitor timeout, are treated as stale/unexpected. A stale
shell/dead/unexpected window is a hard error by default.

After manually confirming a stale or unexpected window is safe to replace:

```bash
REPLACE_STALE=1 scripts/run_i12_pit_0940_strict_shards.sh
REPLACE_STALE=1 scripts/run_i12_pit_0940_sparse_shards.sh
```

If a shard pane is still running the expected Python command but appears hung,
first inspect `pg_stat_activity`, the shard progress artifact, and tmux pane
output. Only then replace it with the separate running-process override:

```bash
REPLACE_RUNNING=1 ONLY_SHARD=may15_21 scripts/run_i12_pit_0940_strict_shards.sh
REPLACE_RUNNING=1 ONLY_WINDOW=sparse_may15_21 scripts/run_i12_pit_0940_sparse_shards.sh
```

`REPLACE_STALE=1` does not kill expected-command Python workers.
`REPLACE_RUNNING=1` is intentionally louder because it can terminate a live
provider replay worker, and the launcher refuses to run it unless exactly one
`ONLY_SHARD` or `ONLY_WINDOW` selector is supplied. Do not use running
replacement broadly after a power loss; inspect the pane and database state,
then replace only the diagnosed shard.

If an old monitor-less pane is present, the launcher will classify it as stale.
Use this recovery flow:

1. Check the pane output and progress artifact.
2. Check Postgres for an idle-in-transaction or wedged provider signature.
3. Replace stale monitor-less panes with `REPLACE_STALE=1`, or kill/relaunch
   pre-monitor panes manually.
4. Replace still-running expected-command panes only with
   `REPLACE_RUNNING=1 ONLY_SHARD=<name>` after diagnosis.

Selector typos fail closed before schema preflight. Valid examples:

```bash
ONLY_SHARD=may15_21 scripts/run_i12_pit_0940_strict_shards.sh
ONLY_WINDOW=strict_may15_21 scripts/run_i12_pit_0940_strict_shards.sh
ONLY_WINDOW=sparse_may15_21 scripts/run_i12_pit_0940_sparse_shards.sh
```

The default sessions/schemas can still be overridden:

```bash
SESSION=i12pit_0940_strict SCHEMA=scratch_i12_pit_m1_0940_strict_20260618 \
  scripts/run_i12_pit_0940_strict_shards.sh
```

## Strict Final Report

```bash
.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_strict_20260618 \
  --report-only \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --minute-path-mode strict_contiguous \
  --report-artifact artifacts/stage0/i12_pit_m1_0940_strict_20260618_report.json
```

## Sparse Final Report

```bash
.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_sparse_20260618 \
  --report-only \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --minute-path-mode sparse_zero_fill \
  --report-artifact artifacts/stage0/i12_pit_m1_0940_sparse_20260618_report.json
```

## Summarize Strict vs Sparse

```bash
.venv/bin/python scripts/summarize_i12_pit_report.py \
  --report artifacts/stage0/i12_pit_m1_0940_strict_20260618_report.json \
  --label strict_0940 \
  --report artifacts/stage0/i12_pit_m1_0940_sparse_20260618_report.json \
  --label sparse_0940
```

For machine-readable output:

```bash
.venv/bin/python scripts/summarize_i12_pit_report.py \
  --format json \
  --report artifacts/stage0/i12_pit_m1_0940_strict_20260618_report.json \
  --label strict_0940 \
  --report artifacts/stage0/i12_pit_m1_0940_sparse_20260618_report.json \
  --label sparse_0940
```

## Resume Missing Strict Dates

The PIT rebuild is date-boundary committed and idempotent by candidate attempt.
To rerun a wedged or missing single date, use the same scratch schema and a
one-day range after the schema already exists. Example for 2026-05-08:

```bash
.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_strict_20260618 \
  --source-hur-schema public \
  --start-date 2026-05-08 \
  --end-date 2026-05-08 \
  --decision-time 09:40 \
  --minute-path-mode strict_contiguous \
  --feed sip \
  --intended-order-usd 250 \
  --max-spread-bps 200 \
  --max-quote-age-seconds 60 \
  --progress-artifact artifacts/stage0/i12_pit_0940_strict_resume_2026-05-08.json
```

To rerun the currently missing strict shard ranges, launch one command per
range:

```bash
for range in 2026-05-08:2026-05-14 2026-05-15:2026-05-21 2026-06-01:2026-06-05; do
  start="${range%%:*}"
  end="${range##*:}"
  safe_start="${start//-/}"
  safe_end="${end//-/}"
  .venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
    --schema scratch_i12_pit_m1_0940_strict_20260618 \
    --source-hur-schema public \
    --start-date "${start}" \
    --end-date "${end}" \
    --decision-time 09:40 \
    --minute-path-mode strict_contiguous \
    --feed sip \
    --intended-order-usd 250 \
    --max-spread-bps 200 \
    --max-quote-age-seconds 60 \
    --progress-artifact "artifacts/stage0/i12_pit_0940_strict_resume_${safe_start}_${safe_end}.json"
done
```

If strict and sparse are ever written into the same scratch schema, use the
runner's compare mode instead:

```bash
.venv/bin/python -m alpha.jobs.run_i12_pit_rebuild \
  --schema scratch_i12_pit_m1_0940_compare_20260618 \
  --report-only \
  --source-hur-schema public \
  --start-date 2026-05-01 \
  --end-date 2026-06-05 \
  --decision-time 09:40 \
  --compare-path-modes \
  --report-artifact artifacts/stage0/i12_pit_m1_0940_compare_20260618_report.json
```
