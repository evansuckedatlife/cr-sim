---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/schedule.py
---

# Reward schedule

How the shaping weight moves over a run, and **only the weight that is real** —
`RewardSchedule`, `KNOBS`, `SHAPING_FIELDS`, `constant_schedule`,
`anneal_to_zero`, driven by `--anneal`.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

**The knob is not the one named "shaping".** `reward_shaping_weight`
(`--shaping`) and the projected potential's shaping are two different things
sharing a word: every `_shaped_value` call site sits inside the `else` of
`if self._reward is not None`, so `--shaping` does nothing at all unless
`--reward simple`. Measured on identical seeds and an identical action stream,
0.01 against 5.00 is bit-identical under `projected` and under `five-term`
(`cr_sim/train/schedule.py:16-28`). A schedule aimed there is a run that
reports an anneal and performs none — which is why the module maps a `--reward`
to the knob that is *actually* its shaping.

`crowns` is never annealed under any knob: that is the objective, not shaping.
At the zero endpoint the episode return equals the final crown difference
exactly, so the schedule terminates on the sparse objective through the same
code path rather than through a special case (`:37-43`).

**Linear, not cosine or exponential**, because the only consumer whose
behaviour depends on the shape is the critic, which carries its scale across
updates — PPO's actor normalises advantages per minibatch and is scale-free,
the value loss fits raw returns and is not. A linear ramp is a constant drift
the critic can track (`:45-51`). **The axis is steps, not updates**, because
`--resume` keeps the step count and replays update indices, and steps is the
one axis a fresh run and a resumed one agree on (`:53-56`).

## Shape

- `SHAPING_FIELDS` `:80` — which fields of each knob an anneal drives to zero;
  everything else in a knob's weight tuple is carried unchanged. `KNOBS` `:87`
  is `tuple(SHAPING_FIELDS)`, so the two cannot drift.
- `_KNOB_FOR_REWARD` `:90`, `knob_for_reward` `:97` — the `--reward` to knob
  mapping.
- `RewardSchedule` `:110`. `start` and `end` are the **full weight tuple
  written out literally at both endpoints**, not a flag plus a delta, so no
  reader has to re-derive either one from a default that may since have changed
  (`:112-115`).
- `constant_schedule` `:252`, `anneal_to_zero` `:257`.
- `end_step = 0` `:126` is the **unset** sentinel, filled in at 80% of the
  run `:180-190`; `shape != "linear"` and an `end_step` before `start_step`
  are refusals in `__post_init__` `:128-154`, not validation a caller may
  skip.
- Resolution and push: `_reward_schedule` `cr_sim/train/run.py:416`, resolved
  once **before** `config.json` is written so the endpoints a reader sees are
  the endpoints the run used `:590-593`. `_push_reward_weights`
  `cr_sim/train/run.py:745-767` sends to **both** the local envs and the parallel pipe,
  under **one** guard — a schedule that has not moved is not pushed, and that
  single condition is what makes a constant schedule cost nothing. Two
  independent conditions guarding one behaviour would mean neither could be
  broken on its own, so no test could hold either to account (`:746-758`).
- Recorded: `config.json`'s `reward_schedule` block carries
  `shaping_is_inert: args.reward != "simple"` `cr_sim/train/run.py:646-649`, and every
  metrics row carries the weights pushed at that update `:779`.

## Connected to

- **owns:** the `reward_weights` field on every metrics row and the
  `reward_schedule` block in `config.json`.
- **owned-by:** [`run-directory.md`](run-directory.md).
- **joins:** [`verdict.md`](verdict.md) — `EVAL_REWARD` exists **because** of
  this module: a probe whose scale follows the training schedule turns the
  promotion criterion into a function of that schedule, and a run's arm shrinks
  against a control that was evaluated once and cached
  (`cr_sim/train/run.py:324-341`, `cr_sim/train/selfplay.py:69-80`).
  Also [`ppo.md`](ppo.md), [`ghost-knobs.md`](ghost-knobs.md).
- **looks-like-but-is-not:** `--shaping`. It is the flag the anneal is *not*
  about under the two rewards anyone trains with, and it is kept as a control.
  See [`ghost-knobs.md`](ghost-knobs.md).

## If you change this

- **Hits:** `config.json`'s key budget — `reward_schedule` is nested precisely
  so it spends two of the four keys `watch.py` allows, not four
  (`cr_sim/train/run.py:640-649`). And both push paths: a weight that reaches `local_envs`
  and not the pipe is bug 1's exact shape, which is why the loop and the RPC
  sit in one function with one guard.
- **Does not hit:** the in-run lift's scale. `EVAL_REWARD` is a constant
  (`cr_sim/train/run.py:342`) and `_eval_env` pins it (`:561`), so annealing the training
  reward does **not** move `eval_lift_sd`. The obvious next assumption — that a
  run under a moving reward reports a moving lift — is what this pinning
  removed, and it is the only reason two runs trained under different rewards
  can be compared at all. Nor does it hit an env's *current* battle: a pushed
  weight is adopted at that env's own next reset, so a row is a target rather
  than a per-battle fact (`cr_sim/train/run.py:774-779`).

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim.train.run --anneal` | the only entry point |
| `cr_sim/api/vec.py` `set_reward_weights` | the worker end of the push (`:244-256`) |
| `cr_sim/train/report.py` | reads `reward_weights` off rows to label a run's scale |
| `tests/test_reward_schedule.py` | holds the inertness claim to account |

## See

- Source: `cr_sim/train/schedule.py`
