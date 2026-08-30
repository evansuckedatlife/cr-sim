---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/measurement/checkpoint.md, objects/interface/check-observation.md, objects/interface/reward-variants.md]
produces: [objects/measurement/verdict.md, objects/measurement/lift.md]
---

# evaluate-against-a-control

Play a saved policy and a control arm over the **same** seed list, difference
them per battle, and divide by the control's own spread.

Four entry points run this movement and share one arithmetic:
`cr_sim/train/evaluate.py:692` (`python -m cr_sim.train.evaluate`),
`scripts/evaluate_checkpoints.py:25`, `scripts/evaluate_vs_expert.py:60`,
`scripts/measure_expert.py:54`. The in-run probes in [`fine-tune.md`](fine-tune.md)
are the fifth spelling and are **not** on the same denominator — see below.

## Input → Movement → Output

In: a checkpoint (or none, for the expert), an episode count, a seed block, an
opponent kind, and a tower level. Movement: draw the block's seeds, play the
control arm, play each mode of the policy over those identical seeds with its
own sampling stream, difference per battle. Out: a `verdict.json` that names who
was played and what was counted, refused if it cannot.

## Why this shape

**Paired seeds, or the effect is under the noise.** The per-episode spread is
several times larger than any effect worth seeing, so both arms play the same
battles rather than the same *number* of battles (`cr_sim/train/evaluate.py:146-151`,
`:330-336`).

**Greedy and sampled are two different policies and are never merged.** The
clone measures +1.623 greedy and +0.709 sampled against one control, so a change
can leave the argmax untouched and move the whole distribution around it
(`:385-392`). Greedy reproduces bit-identically; sampled carries **0.062 sd** of
its own spread, so two sampled readings can differ by 0.17 sd at 95%
(`:157-169`, measured by `scripts/measure_sampled_noise.py`).

**Rotation, not randomisation.** `evaluation_seeds` cuts consecutive disjoint
blocks from one master stream, and `block=0` is byte-identical to the fixed list
every historical number on this project was measured on (`:296-324`). That is
what lets a new reading be read *against* the old ones rather than merely
resemble them.

**A verdict that cannot name its opponent and its reward is refused.**
`write_verdict` (`:483-549`) raises on a missing `eval_opponent` (`:507`), a
missing `eval_reward` where any lift key is present (`:514`), and a file
carrying both a rating and a lift without saying whose lift it is (`:526-546`).

## Steps

1. Load the checkpoint's recorded `observation` and build the environment for
   *that*, not for a flag (`cr_sim/train/evaluate.py:740-742`, `:727-731`). A checkpoint's own
   name for its encoding is the only choice that can be right.
2. `load_policy` (`:116`) routes through `check_observation` (`:89`) — the only
   thing that would catch a v2 checkpoint entering a v1 environment. See
   [`objects/interface/check-observation.md`](../objects/interface/check-observation.md).
3. `evaluation_seeds(episodes, block=...)` (`:757`; `:296`).
4. Build a **fresh environment per arm** (`evaluate_paired:394-397`, `:402`,
   `:460`). One shared environment would share the opponent's generator state
   between arms, which is the thing pairing exists to stop.
5. Play the control: `evaluate(env, None, ...)` (`:403`). The random control
   draws its own actions from `np.random.default_rng(0)` (`:176`), fixed, so the
   control is the same player in every evaluation ever run here.
6. Read `eval_opponent` and `eval_reward` **off the control environment that
   actually played** (`:407`, `:415`; `opponent_name`/`reward_name` at
   `cr_sim/train/selfplay.py:188`, `:163`), never off the flags. A caller cannot
   then label a measurement with an opponent or a scale it did not face.
7. Per mode, build a sampling stream keyed on **which mode this is**, never on
   its index in `modes` (`:437-459`). Keyed on the position, the sampled arm
   drew one stream when both arms were asked for and another when asked for
   alone — the same checkpoint over the same battles measured +1.320 and +1.197,
   decided entirely by whether the caller also wanted greedy. Both single-arm
   callers are live (`scripts/run_ladder.py:381`,
   `scripts/evaluate_vs_expert.py`).
8. `paired_lift` (`:330`): difference per battle, mean over the control's
   `std(ddof=1)`, with a 95% interval off `difference.std(ddof=1)/sqrt(n)`
   (`:352-374`). A single-episode run degrades to a spread of 1 rather than to
   `nan`, because a `nan` propagates into `verdict.json` and reads as a result
   (`:355-360`).
9. Record `control.spread` in the file (`:429-434`). Against a strong opponent
   the control loses every battle the same way, and a small denominator inflates
   every lift measured through it; a reader has to be able to see that.
10. Flatten the headline to whichever arm scored better (`:464-473`), the same
    rule `scripts/clone_policy.py:346` uses, so a verdict written in either place
    means the same thing.
11. `write_verdict` (`:788-789`), and — in the three `scripts/` entry points —
    also a `metrics.jsonl` row through `check_lift_is_named` and a
    `config.json`, so the result lands on the progress page
    (`scripts/measure_expert.py:231-233`,
    `scripts/evaluate_vs_expert.py:224`).

## Two denominators, one word

The four offline paths above use `std(ddof=1)`. The two in-run probes use
`np.std`, i.e. **ddof=0**: `cr_sim/train/selfplay.py:471` and
`cr_sim/train/evaluate.py:635`. An in-run `eval_lift_sd` and an offline `lift`
therefore do not share a denominator even when they share the opponent, the
reward, the seeds and the checkpoint. Recorded on
[`objects/measurement/lift.md`](../objects/measurement/lift.md).

## Two arenas, one unrecorded axis

The four entry points that run this movement do **not** agree on
`--tower-level`, and neither do the eight scripts around them. At 11 the towers
outlast the match and about 90% of battles draw, so the two groups are not the
same game — a verdict from one is not on the other's scale.

Which entry point chooses which, what writes the level down and what checks it
are on
[`../objects/measurement/tower-level.md`](../objects/measurement/tower-level.md),
which is its one home. What matters *here* is the consequence: **nothing in
this movement refuses a verdict that omits the level.** `write_verdict` has
three raises and none of them is this one.

## If you change this

- **Hits:** every historical number. `evaluation_seeds`' block 0 is load-bearing
  (`:310-315`); changing the master seed, the block count, or the draw order
  silently re-bases every comparison in `runs/`.
  [`objects/measurement/verdict.md`](../objects/measurement/verdict.md) and
  [`objects/measurement/metrics-row.md`](../objects/measurement/metrics-row.md)
  — `report.py` reads only the flat keys (`cr_sim/train/report.py:45`, and `:190`).
  [`rate-on-the-ladder.md`](rate-on-the-ladder.md), whose `arms.json` is
  `evaluate_paired` in single-arm mode (`scripts/run_ladder.py:380-381`).
  The in-run probes in [`fine-tune.md`](fine-tune.md), which call `evaluate`
  directly.
- **Does not hit:** the ladder's rating. An Elo is fitted on **crowns**, which no
  reward touches, so a reward change moves every lift here and moves no rating —
  which is why `write_verdict` keys its reward clause to the lift keys and not to
  the file (`_LIFT_KEYS`, `:477-480`). Nor does it hit `report.py`'s chips
  through the nested blocks: only the flattened keys are read.

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | four command lines; the root `CLAUDE.md` rule is to point `--out` at a file for every evaluation worth citing |
| `cr_sim/train/watch.py` | `rglob("*verdict*.json")` across `runs/` **and every worktree's `runs/`** (`cr_sim/train/watch.py:4499`, `_run_roots` `:4427-4440`) |
| `cr_sim/train/report.py` | reads `verdict.json`'s flat keys and withholds `mean_lift`/`best_lift` when a run mixed scales (`cr_sim/train/report.py:54`, `:82-99`) |
| root `CLAUDE.md` | hardcodes the 0.062 sd noise floor and the +1.623 figure |
| `tests/test_measurement.py`, `tests/test_evaluate_decks.py` | hold the refusals and the arithmetic |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `cr_sim/train/evaluate.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`lift.md`](../objects/measurement/lift.md),
  [`verdict.md`](../objects/measurement/verdict.md),
  [`ghost-knobs.md`](../objects/measurement/ghost-knobs.md),
  [`random-streams.md`](../objects/measurement/random-streams.md)
- Source: `cr_sim/train/evaluate.py:137-549`, `:692-791`
