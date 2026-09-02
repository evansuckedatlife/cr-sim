---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/vec.py
---

# VecEnvConfig

The recipe a worker process needs to build its own `CRSimEnv`. Sixteen plain
picklable fields, every one with a default. This is the second of the two
construction paths a training run has, and the one bug 1 lived in.

## Why this shape

**Windows spawns, it does not fork.** There is no copy-on-write parent to
inherit from, so everything a worker touches must survive a pickle across a
pipe. A `Battle` holds a `LogicData` with the whole decoded table set in it;
shipping megabytes of parsed CSV down a pipe on every worker start is wasted
work when the worker can load its own copy from disk. So a worker receives a
build path, two decks and a handful of scalars, and reloads everything itself
(`cr_sim/api/vec.py:5-19`, `_worker` at `:175-177`).

**That is exactly why it is dangerous.** The recipe is written out by hand at
`cr_sim/train/run.py:994-1020`, a second time, next to `_env()`'s hand-written
construction at `:533-546`. Every field has a default
(`cr_sim/api/vec.py:69-113`), so a field set in one and not the other does not
raise — the workers silently take the dataclass default. `tower_level` defaults
to **11** (`:76`), and `--tower-level 5 --workers 8` therefore trained every
rollout at level 11 while `config.json` recorded 5 and the probe evaluated at 5.
At level 11 the towers outlast the match, about 90% of battles draw and crowns
almost never fire, so the agent learned from shaping alone. The headstone is on
the line that fixes it, `cr_sim/train/run.py:999-1008`.

## Shape

- Sixteen fields (`:61-113`): `build`, `blue_deck`, `red_deck`, `team`,
  `ticks_per_second`, `frame_skip`, `level`, `tower_level`,
  `reward_shaping_weight`, `max_ticks`, `reward_weights`, `opponent_seed`,
  `seed`, `net_config`, `shard`, `observation`. Frozen, slotted.
- **Against `CRSimEnv`'s seventeen constructor parameters
  ([`crsim-env.md`](crsim-env.md)), the gap is exactly two:** `skip_forced` and
  `render_mode`. `render_mode` is cosmetic. `skip_forced` is not — it changes the
  MDP, and it is the still-open shape of bug 1.
- The three fields whose absence has each cost a run: `tower_level` (`:76`),
  `observation` (`:113`) — a worker building v1 while the parent's network
  expects v2 is at least a shape error at the first forward, *only because this
  field exists* — and `seed` (`:101`), without which a self-play opponent sampled
  from torch's global stream in a freshly spawned process. Three fresh spawns
  reported `torch.initial_seed()` of 81036942797900, 81144705125800 and
  81234665151700 and shared none of their first twenty opponent actions
  (`:87-100`).
- `_build_env` (`:116-144`) is the single place a field turns into a `CRSimEnv`
  keyword. `observation` is passed through a conditional splat (`:142-143`), so
  `None` means "let `CRSimEnv` default", not "pass `None`".
- `CRSimVecEnv.__init__` (`:297-343`) shards `num_envs` over `workers` and gives
  each worker a **derived** seed, `(seed * 1_000_003 + worker) % 2**31-1`
  (`:327-329`), and a derived opponent seed `opponent_seed + worker * 1000`
  (`:330-333`).
- Five RPC verbs over the pipe (`_worker`, `:147-261`): `reset`, `step`,
  `set_opponent`, `set_reward_weights`, `close`. Terminal episodes are reset
  **inside the worker** and the crown difference comes back with the transition
  (`:216-224`), because resetting in the parent needs another round trip at the
  moment the parent has nothing else to do.
- `WorkerDied` (`:265`) and `_WORKER_TIMEOUT = 120.0` (`:278`): a worker that
  dies takes its traceback with it, so `_recv` (`:345-361`) names the pid and
  tells you to rerun at `--workers 0` rather than hanging.

Verified 2026-08-30 against `main` @ `dc47f51`.

## The parity test is not field-for-field

`tests/test_train.py:564-628` is named
`test_the_worker_config_agrees_with_the_probe_env_field_for_field` and its
docstring says "any field that disagrees means the run measures a different game
from the one it trains on, silently."

**Code wins over that name** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 14). It asserts six `config` fields by hand —
`tower_level`, `ticks_per_second`, `frame_skip`, `max_ticks`,
`reward_shaping_weight`, `reward_weights` (`:599-618`) — plus one behavioural
check that a built worker env reports the right `reward_name` (`:625-628`). It
does **not** iterate `dataclasses.fields`, and it does not assert `observation`,
`seed`, `opponent_seed`, `net_config`, `team`, `level` or the decks. A newly
dropped field is caught only if it happens to be one of those six.

The test does one thing well that is worth keeping: it asserts on the reward the
worker's env is actually **paid**, not on `--shaping`'s transit, because
`--shaping` reaches this config faithfully and is then never read under
`projected` (`:605-612`).

## Connected to

- **owns:** the worker processes' environments, and the two derived seeds.
- **owned-by:** [`../../processes/fine-tune.md`](../../processes/fine-tune.md) —
  the only caller (`cr_sim/train/run.py:991-1023`).
- **joins:** [`crsim-env.md`](crsim-env.md), the other path;
  [`../measurement/reward-schedule.md`](../measurement/reward-schedule.md), whose
  push must reach both (`cr_sim/train/run.py:763-766`);
  [`../measurement/self-play.md`](../measurement/self-play.md) —
  `set_opponent` (`:405-417`) is what lets self-play run in parallel at all,
  worth about three and a half times the throughput;
  [`../measurement/random-streams.md`](../measurement/random-streams.md).
- **looks-like-but-is-not:** a `gymnasium.vector.VectorEnv`. `CRSimVecEnv`
  (`:281-295`) deliberately is not one — that base class exists only when
  gymnasium is installed, which this package does not require. It exposes the
  same batching shape and nothing else.

## If you change this

- **Hits:** `cr_sim/train/run.py:994-1020`, the only place this is constructed,
  and `_env()` at `:533-546`, which must be changed in the same edit.
  `_build_env` (`:116-144`) — a field added here and not turned into a keyword
  there is inert, which is a *quieter* failure than the original bug.
  `tests/test_train.py:564-628`, which must gain an assertion or the new field
  is uncovered.
  [`../measurement/config-json.md`](../measurement/config-json.md) — `config.json`
  still has **no `workers` key**, so nothing in the artifact says which path
  built the environments.
- **Does not hit:** `--shaping`. It is the obvious knob to check when a worker's
  reward looks wrong and it is the **wrong** one: it reaches this config
  faithfully (`:77`) and is then never read under `projected` or `five-term`,
  where 0.01 against 5.00 is bit-identical (`cr_sim/api/env.py:263-281`). The
  field that decides what the workers are paid is `reward_weights` (`:81`).
  Nor does it hit the **card** level: `VecEnvConfig.level` (`:75`) is pinned at
  11 by four independent literals (`cr_sim/api/vec.py:75`,
  `cr_sim/api/env.py:312`, `:659`, `cr_sim/data/card_features.py:164`) and **no
  training or evaluation entry point sets it** — nothing in `cr_sim/train/` or
  `scripts/` has a flag for it, so adding `--level` means touching all four.
  Scope that claim to the training path: `cr_sim/cli.py` **does** have three
  `--level` flags (`:513`, `:518`, `:546`) and `cr-sim battle --level` reaches
  `BattleConfig.level` directly at `cr_sim/cli.py:380`, never through this
  config ([`../surfaces/cli.md`](../surfaces/cli.md)). The two are the same
  field on two construction paths, which is the shape charter row 1 is about.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/run.py` | the only construction site |
| spawned worker processes | rebuild `LogicData`, `LevelTable` and `CardRegistry` per process (`:175-177`), one torch thread each (`:164-173`) |
| `tests/test_train.py` | the parity test, and the `--workers 0` / `--workers N` agreement checks |
| a human debugging | `--workers 0` is the documented way to see a worker's traceback (`:352-353`) |

## See

- Source: `cr_sim/api/vec.py:1-38` (why), `:61-144`, `:147-261`, `:281-461`
- Construction site: `cr_sim/train/run.py:990-1023`
