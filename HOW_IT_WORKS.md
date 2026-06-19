# How Alpha Capital Works (Plain English)

*The $2,000 plan, in simple terms. The numbers below are historical research
ceilings, not live-trading proof. The current blocker is point-in-time validity:
the old I12 corpus selected names using full-day volume, which is not knowable
at the live decision minute.*

---

## What it does, every day

1. **Find candidates through the day** — watch stocks that are far below their
   52-week high and showing intraday stress/volume/quote behavior. The live
   detector must use only data available at that timestamp. Full-day volume is
   forbidden.
2. **Rank as they appear** — a machine-learning model scores candidates at each
   decision time, not only at 09:40.
3. **Maintain a rolling top ~10** — keep the best intended names seen so far,
   then drop any too hard to trade (spread too wide or top-of-book too thin) -> a
   realistic day may hold about **6-8** names.
4. **Buy when the setup confirms** — right after a name qualifies and survives
   the rolling top-K/liquidity rules, split the $2,000 roughly evenly, about
   **$250 each**.
5. **Exit test** — research says next-session open has been better than same-day
   close, but the PIT-clean rebuild should evaluate both. Do not hard-code the
   overnight exit until the clean corpus proves it.

That's the whole thing. A 09:40 run is only one slice of this. The final live
translation is a rolling candidate tape: some names may qualify early, some may
not reveal the volume/price signature until later in the day. At $2k there is
**no borrowing** and no real hedge yet — the one engine is the account until a
second, genuinely different sleeve is built.

> **Current status:** the read-only live machine is built. It can watch, score,
> rank, pull SIP quotes, and log intended trades. It does **not** place orders.
> The current frozen I12 model is loadable, but it is explicitly
> **non-promotable** because its corpus is deferred-PIT. Stage-0 is useful as
> instrumentation; it is not proof that the current model is ready for money.

---

## Daily results (deferred-PIT research ceiling)

| | |
|---|---|
| Average day (mean) | **+1.91%** |
| Median day | +1.20% |
| Biggest win day | +34.2% |
| Biggest loss day | −9.3% |
| Green days | 66% |

## Weekly results (deferred-PIT research ceiling)

| | |
|---|---|
| Average week (mean) | **+9.7%** |
| Median week | +7.1% |
| Biggest win week | +64.1% |
| Biggest loss week | −15.5% |
| Winning weeks | 79% |

> **Cost caveat (measured 2026-06-17):** real Alpaca SIP spread roughly halves the
> old clean-looking compounding on the Jan-Jun 2026 slice. Liquidity filtering
> helps, but it does not fix the PIT issue. Spread realism says the edge might
> survive costs; PIT realism decides whether the live fire set has the edge at
> all.

---

## The honest fine print

- These are **research / historical** numbers from a deferred-PIT corpus. Real
  trading has a harder problem: the detector has to fire using only live
  information.
- The current I12 model freeze is mechanically loadable but not production-clean:
  `stage1_i12_403a5ae359cd_accecdda` is `shadow` and non-promotable with
  `deferred_pit_model`.
- The next serious model is a PIT-clean rolling rebuild. It should generate
  candidate rows at multiple intraday decision times, train only on
  as-of-timestamp features, maintain a rolling top-K, and compare same-day close
  versus next-session open.
- These stocks are **thin** (small, low-volume). This works at small size; it cannot scale to
  big money.
- We run **unlevered** (no borrowing) — these stocks can't be margined at $2k anyway.
- The risk is real: a bad **week** can be about **−15%**, and a rough multi-week stretch could
  draw the account down roughly **−27%**. The fix — a second, *different* strategy that wins
  when this one loses — is future work, not part of the $2k plan.

## If The PIT-Clean Rebuild Looks Weak

The next move is not "add a neural net." The next move is to ask whether the
live-visible candidates contain an early signature that separates future winners
from future losers.

Do the simple rebuild first. If it underperforms, study the names the live
screen would actually have seen at each decision time, starting with
9:35/9:40/9:45/10:00 and extending later if the old winners reveal volume later:

- volume curve and acceleration
- early price path and stabilization
- spread and top-of-book behavior
- catalyst/hazard tags
- same-day exit versus next-open exit
- first qualifying timestamp and whether later qualifiers contain the lost edge

The important rule: compare winners versus losers **inside the live candidate
set**. Do not study only old full-day-volume winners. If a useful early shape
exists, turn it into explicit engineered features for the current GBRT. A
raw-tape neural net is only worth testing after that simpler model plateaus
out-of-time.

## What Stage-0 Proves

Stage-0 is not a profit test. With the current deferred-PIT model, it answers
one narrow instrumentation question:

> When the live candidate machinery selects names, are the quotes current and
> are the names actually tradeable at about $250 each?

It logs each intended trade with the live quote, spread, top-of-book size, halt
condition inference, skipped reason, same-day/exit evidence when available, and
next-session exit quote. Names skipped for spread/size/halt count as cash.
Nothing is bought.
