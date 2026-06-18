# Alpha Capital Documentation

This repository contains the runnable code. The wider strategy and operations
canon may also live in the Alpha Capital vault:

```text
~/Documents/AlphaCapital/
```

The vault is intentionally more detailed than this repo, but older vault files
can be stale. When docs disagree, trust the runnable repo, migrations, tests,
and current checkpoint first.

## Current Navigation

- Root `README.md` - repo/system map and current engineering posture.
- `POWER_OUTAGE_CHECKPOINT.md` - immediate recovery state and exact run order.
- `Present Task List.md` - current task spine; old task prompts are superseded
  unless explicitly revived.
- `HOW_IT_WORKS.md` - plain-English $2k/I12 plan and current caveats.
- Vault `Strategy-State-2026-06.md` - strategy source of truth when present.
- Vault `Architecture.md` - system/infra source of truth when present.

## Repo-Facing Rule

Repository docs should describe the intended production standard and the current
operational state only when it is audit-backed and expected to remain useful.

When code behavior changes in a way that alters a pattern contract, update the
relevant vault or repo-facing doc in the same change. The README should stay
concise; detailed doctrine belongs in the vault or a focused design doc.

## Current Caution

As of 2026-06-17, I12 Stage-0 read-only fill-test code exists and Alpaca SIP
quotes are entitled. The frozen I12 model is loadable for read-only Stage-0
plumbing only:

```text
model_id: stage1_i12_403a5ae359cd_accecdda
status: shadow
promotability: non-promotable, deferred_pit_model
```

The old I12 research corpus used a full-day volume selector. That is future
information at the live decision minute, so the current model is a research
upper bound and teacher, not a production trading model. The next production
gate is a PIT-clean I12 rebuild using only as-of-entry intraday evidence, with
same-day-close and next-session-open exits both evaluated.

If the simple PIT-clean live-visible set is weak, the follow-up is a diagnostic
study of winners versus losers within that same live-visible set, then explicit
early-curve features for the current GBRT. Do not jump directly to a neural net
or study only old full-day-volume winners.

Do not use old 17-pattern/KOTH planning docs to steer current work, and do not
promote any deferred-PIT I12 model as production-clean.
