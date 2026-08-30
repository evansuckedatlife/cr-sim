---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/interface/crsim-env.md, objects/interface/vec-env-config.md, objects/measurement/checkpoint.md, objects/measurement/reward-schedule.md]
produces: [objects/measurement/run-directory.md, objects/measurement/checkpoint.md, objects/measurement/metrics-row.md]
---

# fine-tune

Run PPO against a fixed or self-playing opponent, probe the policy every
`--eval-every` updates, and promote on a rolling mean of whatever statistic
that probe returns.

`python -m cr_sim.train.run`, `cr_sim/train/run.py:510`. Self-play is not a
separate verb: it is `--opponent self`, which swaps who the environment faces
and adds two probes. Everything else about the movement is identical.

## Input → Movement → Output

In: forty flags (`build_parser`, `:82-347`), optionally a `cloned.pt` through
`--init-from`, optionally an offline ratings table through `--ladder-ratings`.
Movement: build `--envs` environments — locally, or in `--workers` processes
through `VecEnvConfig` — collect `--horizon` steps each, compute GAE, run the
PPO update, then every `--eval-every` updates play a probe and maybe write
`best.pt`. Out: `runs/<name>/` with `config.json` written once up front,
`metrics.jsonl` appended per update, and three checkpoints.

## Why this shape

**`config.json` is written before the first update** (`:605-654`) so that a run
directory records what the run was *asked* to be. That is the only reason bug 1
was ever findable: the file said `tower_level: 5` while every `--workers`
rollout ran at 11.

**The network cannot exist before the first observation**, so `train` builds it
and hands it back through `on_net` (`cr_sim/train/ppo.py:206-209`, `:243`;
`_on_net` at `cr_sim/train/run.py:935`). Everything that needs weights — the opponent pool,
the probes, the workers' first opponent — is wired inside that callback.

**Promotion is on a rolling mean, never a single reading** (`:829-865`). Keeping
the highest lift is keeping the luckiest: measured, the checkpoint chosen that
way scored +0.375 on its 40 battles and -0.033 on 300, while the final weights,
chosen by nothing at all, scored +0.141. And what it promotes *on* is the
ladder's Elo where there is one, because `eval_lift_sd` is the sampled arm at
forty battles and is the noisiest number in the run.

**The rolling key is never called `lift`.** An Elo and a lift are unrelated
scales and this project has paid three rounds of invalid comparisons for putting
two scales under one name (`:858-864`; `promoted_on` recorded into `best.pt` at
`:881`).

## Steps

1. Resolve `anchor_path = --kl-reference or --init-from` **before** anything is
   written, because `config.json` records it (`:513`). Refuse `--kl > 0` with no
   anchor: anchoring to a random initialisation is not a trust region
   (`:669-673`). Refuse `--init-from` together with `--resume` (`:674-679`), and
   refuse an `--init-from` whose recorded `head` differs from `--head`
   (`:684-689`).
2. Resolve the device (`_resolve_device`, `:350`). `auto` never chooses `xpu`
   and says why in place (`:363-385`) — see
   [`objects/measurement/ghost-knobs.md`](../objects/measurement/ghost-knobs.md).
3. Define `_env()` (`:533-546`) — the **local** construction path. It sets
   observation, TPS, frame skip, max ticks, `tower_level`,
   `reward_shaping_weight`, `reward_weights` and the opponent, and nothing else.
4. Define `_eval_env()` (`:548-561`): a **random** opponent at seed 90 000, and
   `EVAL_REWARD` rather than the training reward. Both halves matter — the
   control wins 92% of idle matches and 26% of random ones, and a probe whose
   scale follows the training schedule turns promotion into a function of that
   schedule (`:330-342`).
5. Resolve the reward schedule once (`_reward_schedule`, `:416-435`; resolved at
   `:593`) so `config.json` records the endpoints the run used rather than flags
   a reader has to re-derive. See
   [`objects/measurement/reward-schedule.md`](../objects/measurement/reward-schedule.md).
6. Write `config.json` (`:605-654`). Every key here is spent out of the progress
   page's four-key A/B budget, and the comments at `:635-651` account for the
   spend in place. See
   [`objects/measurement/config-json.md`](../objects/measurement/config-json.md).
7. Build the parallel environments if `--workers` (`:990-1023`). This is the
   **second** construction path, and every field `_env()` sets must be set here
   too — the comment at `:999-1008` is bug 1's headstone, sitting on the
   `tower_level=args.tower_level` line it was missing.
8. `train(...)` (`cr_sim/train/ppo.py:166`, called at `cr_sim/train/run.py:1026-1044`):
   - seed torch and two numpy streams (`cr_sim/train/ppo.py:211-222`) — the minibatch
     shuffler is its own generator because `np.random.shuffle` draws from
     numpy's *global legacy* `RandomState`, which nothing seeds;
   - build the network **after** seeding (`:243`);
   - collect `--horizon` steps (`:311-355`), stepping either the vec env
     (`:322`) or the local envs (`:331-345`);
   - GAE (`compute_gae`, `:113`; called `:361-364`), then `_update`
     (`:464`, called `:366`);
   - every `refresh_every` updates fire `on_refresh` **then** refresh the
     snapshots, in that order, so the pool contains this generation before the
     opponents draw from it (`:377-389`);
   - attach `noop_fraction`, `ret_mean`, `ret_std` and `explained_variance`
     (`:396-420`) and call `on_update`.
9. `record` (`cr_sim/train/run.py:769`) runs per update, in this order: push the schedule's
   weights at this **step** (`_push_reward_weights`, `:745-767`) — to the local
   envs *and* over the workers' pipe, both always, because a field one path sets
   and the other does not is bug 1 exactly (`:760-766`); stamp
   `reward_weights` onto the row (`:779`); print; then the probes.
10. The probes, every `--eval-every` updates: `ancestor_probe` under
    `--opponent self` (`:803-808`; `cr_sim/train/selfplay.py:488`), then one of
    `ladder_probe` / `rotating_probe` / `evaluation_probe`
    (`:969-988`). `rotating_probe` is **built but not the default**
    (`:979-984`); see its own adoption note at
    `cr_sim/train/evaluate.py:592-600`.
11. Promote (`:851-885`) and periodically checkpoint (`:886-905`).
    `checkpoint.pt` carries the optimiser state deliberately (`:887-890`);
    `best.pt` and `final.pt` do not.
12. `_write` last, never on the way in (`:908-914`). The probe adds its fields
    to the same dict, so writing first recorded every row without the one number
    worth keeping. Every row goes through `check_lift_is_named` (`:913`).
13. `final.pt` after the `with` block closes (`:1053-1054`).

## If you change this

- **Hits:** both construction paths, always.
  [`objects/interface/crsim-env.md`](../objects/interface/crsim-env.md) and
  [`objects/interface/vec-env-config.md`](../objects/interface/vec-env-config.md)
  — a field added to one and not the other is bug 1, and the divergence is
  silent because `VecEnvConfig` has defaults for everything
  (`cr_sim/api/vec.py:69-113`).
  [`objects/measurement/config-json.md`](../objects/measurement/config-json.md)
  — a new key costs a slot in `_AB_MAX_DIFF` (`cr_sim/train/watch.py:1799-1803`).
  [`objects/measurement/metrics-row.md`](../objects/measurement/metrics-row.md)
  — the row is an open dict with one guard.
  [`objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md) —
  three payload shapes, and `HEAD_BY_PARAMETERS` (`cr_sim/train/ladder.py:98`) is what reads
  a checkpoint that records neither `head` nor `observation`.
  `scripts/register_job.py` conventions, because the run directory this writes
  is the same shape a job writes.
- **Does not hit:** `CRSimSelfPlayEnv` (`cr_sim/api/env.py:649`). It is the
  obvious next stop when the word is "self-play" and it is the **wrong** one:
  `--opponent self` builds ordinary `CRSimEnv`s facing a `PooledOpponent`
  (`:577-585`, `:947-966`), and `CRSimSelfPlayEnv` takes no `reward_weights` and
  no `observation` at all, so nothing in this process can reach it.
  Nor does it hit `--shaping` under `projected` or `five-term`: every
  `_shaped_value` call site is in the branch those rewards do not take, and 0.01
  against 5.00 is bit-identical under both (`cr_sim/api/env.py:263-281`).

## The shape of bug 1 that is still open

`CRSimEnv.__init__` accepts `skip_forced` (`cr_sim/api/env.py:318`, default
`True`, read at `:462`). `VecEnvConfig` has **no such field**
(`cr_sim/api/vec.py:69-113`) — it is the only behavioural environment parameter
the worker recipe cannot carry. No entry point sets it today, so nothing is
wrong now; the moment a `--skip-forced` flag is added to `run.py` and wired
through `_env()` alone, bug 1 reproduces exactly, with `config.json` recording
the flag and every `--workers` rollout ignoring it.

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | `python -m cr_sim.train.run --name ... --workers 8` |
| `cr_sim/train/watch.py` | enumerates `runs/*/metrics.jsonl` and renders everything here |
| `cr_sim/train/report.py`, `cr_sim/train/notify.py` | read the run directory |
| worker processes | `multiprocessing.spawn` children; attribute them to their parent pid before concluding anything is orphaned (root `CLAUDE.md`) |
| `tests/test_train.py`, `tests/test_trust_region.py`, `tests/test_selfplay_pool.py` | hold the flag wiring, the KL anchor and the pool |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `cr_sim/train/evaluate.py`, `cr_sim/train/ladder.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`ppo.md`](../objects/measurement/ppo.md),
  [`self-play.md`](../objects/measurement/self-play.md),
  [`reward-schedule.md`](../objects/measurement/reward-schedule.md),
  [`random-streams.md`](../objects/measurement/random-streams.md),
  [`explained-variance.md`](../objects/measurement/explained-variance.md),
  [`run-directory.md`](../objects/measurement/run-directory.md)
- Source: `cr_sim/train/run.py:510-1057`, `cr_sim/train/ppo.py:166-461`
- As-built: `docs/training.md`
