---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/env.py
---

# CRSimEnv

One battle, viewed from one team's side, as a single-agent environment. The only
environment `train/run.py`, `train/evaluate.py`, `train/ladder.py`,
`scripts/clone_policy.py`, `scripts/make_demos.py` and `api/vec.py` build.

## Why this shape

It is the **seam** where a battle stops being a state and becomes a promise
about a tensor's shape, and it is the object whose seventeen constructor
keywords are the whole surface a run configures. Two things about it are
load-bearing and neither is obvious from the signature:

**Every parameter is a keyword with a default** (`:307-319`), so a caller that
forgets one gets a working environment playing a different game. That is bug 1's
mechanism, one object down: nothing raises.

**A reward change is adopted at `reset`, never mid-episode** (`set_reward_weights`,
`:418-442`; adopted at `:402-407`). Mid-episode the reward stops being
potential-based — the tracker's previous potential is under the *old* weight, so
the next step is paid a genuine reward plus a fabricated one for the weight
change, charged in full to whatever action happened to be taken there. Measured,
switching `ProjectionWeights` from (tower=1, elixir=0.3) to (0, 0) at step 5:
-0.007802 without the switch, -0.159656 with it. Nineteen times the genuine
reward, on one arbitrary action — **and invisible in aggregate**, because the
episode return still telescopes to its own endpoint weights and the telescoping
test stays green over it.

## Shape

- `__init__` takes 3 positional and 14 keyword parameters
  (`cr_sim/api/env.py:300-320`): `data`, `levels`, `registry`, `blue_deck`,
  `red_deck`, then keyword-only `team`, `opponent_policy`, `ticks_per_second`,
  `frame_skip`, `level`, `tower_level`, `reward_shaping_weight`,
  `reward_weights`, `max_ticks`, `render_mode`, `skip_forced`, `observation`.
- **`level` and `tower_level` are different ladders** and both default to 11
  (`:312-313`). See the collision table in `../../CONTEXT.md`; do not restate it.
- `reward_weights`'s **type** selects the reward — `None` is meaningful and
  selects the simple shaped one (`:336-341`, `_build_reward` at `:341`). That is
  why `_pending_reward` is a zero-or-one-element list rather than an optional
  (`:342-346`).
- `.encoding` (`:372`) is the single `EncodingConfig` a policy network's shapes
  are read from — see
  [`encoding-config.md`](encoding-config.md) and [`net-config.md`](net-config.md).
- `reset` (`:382-416`) builds a fresh `Battle` from the seed; with `seed=None` it
  draws one from an unseeded `np.random.default_rng()` (`:385`).
- `step` (`:444-495`) applies the action, lets the opponent act, advances
  `frame_skip` ticks, scores, and then — if `skip_forced` — runs out every
  following state that has one legal action.
- `info` carries `hash`, `tick`, `blue_crowns`, `red_crowns`, `finished`,
  `reason` (`_info`, `:229-242`). The hash is what a determinism check compares.
- `_apply_action` (`:168-184`) deliberately does **not** consult the mask, so the
  mask and `play_card` are two independent legality checks that cannot drift.
  `_opponent_move` (`:514-516`) and the run-out (`:562`) *do* trust the mask,
  taking `argwhere(mask)[0]` without re-validating.

Verified 2026-08-30 against `main` @ `dc47f51`.

## `skip_forced` — the still-open shape of bug 1

`skip_forced` (`:318`, stored `:349`, read `:462`) changes the MDP the agent
sees and which reward-accounting path runs. About 89% of decisions at the
default cadence have exactly one legal action (`:531-537`), so running them out
is roughly nine times the useful samples per step — and a telescoping reward is
then scored once at the end of the run-out rather than at every state passed
through, which was 2.00 score calls per decision and 26.4% of all environment
wall time (`:466-480`, `:549-555`). The loop is bounded by
`_MAX_FORCED_RUN_OUT = 4096` (`:147-151`).

**It has no `VecEnvConfig` field** (absent from `cr_sim/api/vec.py:69-113`) and
no CLI flag anywhere. Under `--workers` it can therefore never be anything but
`True`. Only `tests/test_train.py:225`, `:252`, `:270` ever set it. Nothing is
wrong today; the moment someone adds `--skip-forced` and wires it through
`run.py`'s `_env()`, bug 1 reproduces with `config.json` recording the flag and
every worker rollout ignoring it.

## Connected to

- **owns:** the `Battle` it resets (`:386-399`), its `EncodingConfig`
  (`:356-357`), and the pending-reward boundary.
- **owned-by:** [`../../processes/fine-tune.md`](../../processes/fine-tune.md),
  [`../../processes/collect-demonstrations.md`](../../processes/collect-demonstrations.md),
  [`../../processes/evaluate-against-a-control.md`](../../processes/evaluate-against-a-control.md)
  — every verb builds one.
- **joins:** [`vec-env-config.md`](vec-env-config.md), the second construction
  path; [`observation-features.md`](observation-features.md);
  [`action-mask.md`](action-mask.md);
  [`reward-variants.md`](reward-variants.md);
  [`../measurement/reward-schedule.md`](../measurement/reward-schedule.md), which
  pushes through `set_reward_weights`.
- **looks-like-but-is-not:** `CRSimSelfPlayEnv` (`:610`). **Leftover**, and its
  own docstring says so: "nothing but `tests/test_api_env.py` builds this today,
  `cr_sim.train.run` does self-play through `CRSimEnv`'s `opponent_policy`
  instead" (`:639-646`). It takes eleven parameters (`:649-660`) and **neither
  `reward_weights` nor `observation` is among them** — `build_encoding_config` is
  called with no observation at `:677`, so it is hardcoded to v1, and any
  comparison against `CRSimEnv` is a comparison across observations. It can never
  carry a reward schedule, and
  [`../measurement/reward-schedule.md`](../measurement/reward-schedule.md) names
  that as deliberate rather than missing.

## If you change this

- **Hits:** [`vec-env-config.md`](vec-env-config.md) **always**. A new keyword
  here with no field there is silently the dataclass default in every worker
  process. And the parity test will not catch it — see that card.
  `cr_sim/train/run.py:533-546` (`_env`) and `:994-1020` (the `VecEnvConfig`
  literal), the two paths that must be edited together.
  [`../measurement/config-json.md`](../measurement/config-json.md) if the knob is
  worth recording, at the cost of a slot in the A/B budget.
  Every `scripts/` entry point that builds one — eight of them, each with its own
  argparse defaults, and they already disagree about `tower_level`.
- **Does not hit:** `Battle` or anything in `battle/`. This class configures a
  battle; it does not change one. A tick-phase or entity change is
  [`../battle/battle.md`](../battle/battle.md), and it reaches here only through
  the state hash in `info`.
  It also does not hit the reward *mid-episode*: `set_reward_weights` stages, and
  `reset` adopts. A caller that wants an immediate change is asking for the
  fabricated-reward bug measured at `:426-432`.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/run.py` | builds two ways: `_env()` locally, `VecEnvConfig` in workers |
| `cr_sim/train/evaluate.py`, `ladder.py`, `selfplay.py` | build one per arm, deliberately not shared |
| `scripts/{make_demos,clone_policy,run_ladder,evaluate_checkpoints,evaluate_vs_expert,measure_expert,evaluate_decks,measure_sampled_noise}.py` | eight independent constructions with eight sets of argparse defaults |
| `cr_sim/play/server.py` | builds one for the browser, with its own deck and its own `--tower-level` default of 11 (`cr_sim/play/server.py:46`, `:306`) |
| `tests/` | ~30 files build one through a `world` fixture |

## See

- Source: `cr_sim/api/env.py:245-607`
- As-built: `docs/training.md`
