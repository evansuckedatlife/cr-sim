---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/evaluate.py
---

# Lift callers — every site that computes one, and in which denominator

The blast radius of `paired_lift`. [`lift.md`](lift.md) owns what a lift *is*;
this card owns *who computes one*, because a list of call sites maintained in
prose on another card drifts the next time a script is added — and it had, by
two files and four sites.

## Why this shape

Editing the arithmetic is a change-impact question with a waterfall too long
for one bullet, and the four sites that do **not** call `paired_lift` matter as
much as the six that do: they will not move with it, and one of them produced
the noise floor every battle count in this project is sized off. Three
denominator conventions coexist and none of them raises.

## Shape

**Three conventions, one formula.**

| Denominator | Where | Who reads it |
|---|---|---|
| `std(ddof=1)`, through `paired_lift` | `cr_sim/train/evaluate.py:359` | every offline path |
| `std(ddof=1)`, spelled by hand | `scripts/clone_policy.py:322`, `scripts/evaluate_checkpoints.py:55` | two scripts that never import it |
| `np.std`, **ddof=0** | `cr_sim/train/selfplay.py:471`, `cr_sim/train/evaluate.py:635` | the two in-run probes, and therefore `eval_lift_sd` and every promotion decision |

So an in-run `eval_lift_sd` and an offline `lift` do not share a denominator
convention even when they share the control, the reward, the arm, the episode
count, the arena and the seed block.

**The six `paired_lift` call sites, in four files.**

| Site | What it produces | Notes |
|---|---|---|
| `cr_sim/train/evaluate.py:462` | the per-mode block of a `verdict.json` | inside `evaluate_paired`; the flattened headline keys come off it at `:465-473` |
| `scripts/measure_expert.py:146` | the expert's verdict | the source calls it "the one arithmetic, not a fourth spelling of it" (`:143-145`) |
| `scripts/evaluate_decks.py:325` | a per-deck row, plus its own `control_spread` at `ddof=1` (`:339`) | filed **leftover**: one closed question about `FactoredStatsHead` |
| `scripts/measure_sampled_noise.py:76`, `:78` | greedy, twice, asserted **bit-identical** (`:82-85`) | the assertion is what makes the next row about sampling and not about seeds |
| `scripts/measure_sampled_noise.py:89` | one lift per sampled stream; their `std(ddof=1)` at `:98` **is the 0.062 sd noise floor** | ran once; the figure is quoted in root `CLAUDE.md:67-77` and `cr_sim/train/evaluate.py:157-167` |

That last row is why this card exists. `scripts/measure_sampled_noise.py` is
filed `leftover` and has no card of its own, so it fell out of a hand-written
Hits list — and it is the script that produced the number every battle count in
this project is sized off. Change the formula without re-running it and the
floor is silently re-based against the lifts it is the floor *for*.

Citations: `cr_sim/train/evaluate.py:330`, `:359`, `:462`, `:465-473`, `:635`;
`cr_sim/train/selfplay.py:471`; `scripts/measure_expert.py:143-146`;
`scripts/evaluate_decks.py:325`, `:339`; `scripts/measure_sampled_noise.py:76`,
`:78`, `:82-85`, `:89`, `:98`; `scripts/clone_policy.py:322`, `:329`;
`scripts/evaluate_checkpoints.py:55`, `:91`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing. It is the caller index for one function.
- **owned-by:** [`lift.md`](lift.md).
- **joins:** [`verdict.md`](verdict.md) (five of the six sites end in one),
  [`metrics-row.md`](metrics-row.md) (the probes do not),
  [`random-streams.md`](random-streams.md) (`measure_sampled_noise` and
  `evaluate_decks` each construct an explicit `torch.Generator`).
- **looks-like-but-is-not:** `fit_ratings`. Elo is not a lift and never shares
  its axis ([`ladder.md`](ladder.md)).

## If you change this

- **Hits:** all six sites, in four files. The two `ddof=1` hand-copies and the
  two `ddof=0` probes will **not** move, so a formula change widens the gap
  between conventions rather than closing it. Re-run
  `scripts/measure_sampled_noise.py` in the same change or the 0.062 floor no
  longer describes the arithmetic it is quoted beside.
- **Does not hit:** anything on disk. Every recorded lift stays on the scale it
  was measured on, which is why nothing here is retro-fitted — and why the
  obvious next step, "recompute the old verdicts under the new formula", would
  destroy the only record of what they meant.

## Surfaces

| Surface | Role |
|---|---|
| `runs/*/verdict*.json` | written by five of the six sites |
| `runs/sampled-noise-floor/noise.json` | written by the sixth |
| root `CLAUDE.md:67-77` | quotes the floor as authoritative |
| `tests/test_measurement.py`, `tests/test_evaluate_decks.py` | pin two of the four files |

## See

- Source: `cr_sim/train/evaluate.py`
