# You are about to quote or compare a number — bug 6

**Read this before you write the sentence, not before you write the code.**
Everything in [`CONTEXT.md`](CONTEXT.md) is keyed on an edit: eleven entries,
every one of them beginning "Changing…" or "Adding…". Bug 6 was not committed by
an edit. It was committed by someone quoting two numbers that already existed,
in a sentence, with no code changed and nothing to run — and a map keyed
entirely on write-time arrivals cannot catch a read-time bug.

That cost three rounds of invalid comparisons and one retracted headline (root
`CLAUDE.md:56-63`).

---

## The refusal checklist

A lift is a function of **six** things
([`../objects/measurement/lift.md`](../objects/measurement/lift.md)). Two
numbers may not share an axis, a chart, a sentence or a field until **all six
match**. Not "probably match" — until you have read all six off both.

| # | Input | Where you read it off a number | If it is absent |
|---|---|---|---|
| 1 | **the control** | `eval_opponent` on the row or in the verdict | `write_verdict` and `check_lift_is_named` refuse it — so if it is missing, the number came from a writer that skips them |
| 2 | **the reward** | `eval_reward`; the two literal scales are `simple:shaping=0.01` and `projected:tower=1,elixir=0.3,horizon_seconds=3` | same two guards refuse it. The reward is in the numerator **and** the denominator |
| 3 | **the arm** | greedy or sampled. Greedy reproduces bit-identically; sampled carries **0.062 sd** of its own | never merged in code, routinely merged in prose |
| 4 | **the episode count** | `episodes` | a small count is not a smaller version of a big one: `ddof=1` is undefined on one episode and the spread falls back to `1.0` (`cr_sim/train/evaluate.py:355-359`) |
| 5 | **the arena** | `tower_level` | **nothing refuses a verdict that omits it**, and it usually is omitted — [`../objects/measurement/tower-level.md`](../objects/measurement/tower-level.md) |
| 6 | **the seed block** | `block` / `eval_block` | block 0 is byte-identical to the list every historical number sits on; a different block is a different sample |

## Three ways two numbers differ with nothing raising

Each is live in the tree today, and none of them produces an error, a warning or
a failing test.

1. **The denominator.** Offline paths divide by `std(ddof=1)`; the two in-run
   probes divide by `np.std`, ddof=0. Ten sites, three conventions —
   [`../objects/measurement/lift-callers.md`](../objects/measurement/lift-callers.md).
   So an in-run `eval_lift_sd` and an offline `lift` are on different scales
   **even when all six inputs above match**.
2. **The arena.** The twelve entry points split between two tower levels, one
   program writes it into a verdict and one reads one back — the table is on
   [`../objects/measurement/tower-level.md`](../objects/measurement/tower-level.md),
   which is its one home.
3. **The family.** An Elo and a lift are not the same kind of number and never
   share an axis or a field: Elo is fitted on crowns, which no reward touches.
   `write_verdict` refuses a file carrying both without saying whose lift it is;
   `run.py` refuses to call a rolling Elo `rolling_lift`
   ([`../objects/measurement/verdict.md`](../objects/measurement/verdict.md),
   [`../objects/measurement/ladder.md`](../objects/measurement/ladder.md)).

## Where the guards already stop you, and where they do not

**They do stop you** at `check_lift_is_named` (`cr_sim/train/selfplay.py:55`,
five raises) and `write_verdict` (`cr_sim/train/evaluate.py:483`, three raises),
and at `report.collect`, which withholds `mean_lift` and `best_lift` **entirely**
when a run mixed scales rather than averaging across them
(`cr_sim/train/report.py:82-99`). If one of those refuses your number, that is
the guard working — do not route around it.
`scripts/register_job.py:50` is the one metrics writer that skips them, and
`cr_sim/train/watch.py:526-528` names it as the exception in source.

**They do not stop you** anywhere a person reads a number and types it
somewhere else. Point `--out` at a file for every evaluation worth citing: a
number transcribed by hand is a number nobody can check, and that is how
`+1.813` entered the record for a figure that is `+1.623`
(root `CLAUDE.md:88-90`).

## And size the comparison before you make it

The sampled noise floor is **0.062 sd**, not the ±0.02–0.04 that older writing
here quotes. Two sampled runs can differ by **0.17 sd** at 95% while measuring
nothing but the generator. Anything sized off the old figure — battle counts
especially — is off by 4x (root `CLAUDE.md:67-77`;
`cr_sim/train/evaluate.py:157-167`; the measurement is
`runs/sampled-noise-floor/noise.json`).

## Then

If all six match and the denominators agree, say so **in the sentence**. A
comparison whose scale is stated can be checked by the next reader; one whose
scale is implied cannot, and that is the whole of bug 6.

## See

- [`../objects/measurement/lift.md`](../objects/measurement/lift.md) — what a
  lift is
- [`../objects/measurement/verdict.md`](../objects/measurement/verdict.md) — the
  two guards
- [`../objects/measurement/tower-level.md`](../objects/measurement/tower-level.md)
  — the input with no guard
- [`../processes/evaluate-against-a-control.md`](../processes/evaluate-against-a-control.md)
  — how to produce one properly instead
