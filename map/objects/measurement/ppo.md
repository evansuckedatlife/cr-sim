---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/ppo.py
---

# PPO

The algorithm, kept importable without argument parsing, file layout or
checkpoint policy — `PPOConfig`, `Rollout`, `train`, `_update`, `compute_gae`.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

Separated from `cr_sim.train.run` so the algorithm stays testable without
dragging a run directory in behind it (`cr_sim/train/run.py:3-6`). Two of its
knobs are shaped by measurements taken here rather than by convention:
`value_learning_rate` is raised above the actor's because the critic is solving
a plain regression while the actor is walking a trust region, and tying them
holds the critic back for the actor's benefit (`cr_sim/train/ppo.py:62-68`);
`kl_coefficient` is `KL(reference || policy)` and the **direction is
deliberate** — that is the one that punishes dropping what the reference does,
and the reverse is mode-seeking and happily collapses onto one action, which
here means passing (`:80-92`).

The seeding is the part with a bug history. `_update` takes its `shuffler`
**required, not defaulted**, because a default would be a second mechanism
guarding one behaviour: `train` could stop passing its own and every run would
stay reproducible while quietly replaying one permutation, which no single-line
mutation could then be held to account for (`:474-483`).

## Shape

- `PPOConfig` `:54`, `Rollout` `:98`, `compute_gae` `:113`, `train` `:166`,
  `_update` `:464`.
- Three streams, all owned: `torch.manual_seed(config.seed)` `:211` (the
  learner's own sampling, this process only), `rng = default_rng(config.seed)`
  `:212` (episode resets), `shuffler = default_rng(config.seed + 1)` `:222` —
  which replaced `np.random.shuffle` on numpy's **global legacy RandomState**,
  seeded from OS entropy at import and by nothing here `:213-221`.
- Environment seeds are `config.seed * 1000 + index` `:233`, `:239`, so
  identical seeds across the batch cannot collect the same battle several
  times.
- **`PPOConfig.gamma` `:69` is unreachable.** No CLI flag anywhere sets it;
  `cr_sim/train/run.py:594-603` constructs `PPOConfig` without it, and `clone.collect`'s own
  `gamma` `cr_sim/train/clone.py:245` has no caller passing it either. Both are
  *unreachable* knobs, not dead code — the value is used
  (`cr_sim/train/ppo.py:137-138`, `cr_sim/train/clone.py:393`, `:504`), it just cannot be changed from
  outside.

## Connected to

- **owns:** the `Rollout` the update consumes and every loss field on a metrics
  row.
- **owned-by:** [`run-directory.md`](run-directory.md) — `run.py` owns where
  the numbers go; this owns how they are produced.
- **joins:** [`random-streams.md`](random-streams.md) (the shuffler is one of
  the five closed in `8fbe4a5`); [`checkpoint.md`](checkpoint.md) (the
  optimiser state `checkpoint.pt` carries is this module's);
  [`self-play.md`](self-play.md) (`opponents` and `refresh_every` are handed in
  from there).
- **looks-like-but-is-not:** the reward schedule. `train` never reads a
  schedule; `run.py`'s `record` callback pushes weights to the environments at
  the top of every update (`cr_sim/train/run.py:769-773`), so from PPO's side the reward is
  just whatever the env returned. See
  [`reward-schedule.md`](reward-schedule.md).

## If you change this

- **Hits:** every recorded run's comparability. The shuffler seed, the env seed
  arithmetic and `torch.manual_seed` together define what `--seed 0` means; a
  change to any of them makes a re-run of an old command a different
  experiment with the same name in `config.json`.
- **Does not hit:** the two probes or the evaluation. `evaluate` builds its own
  environments and its own generators and never touches PPO's
  (`cr_sim/train/evaluate.py:137`), so a change here moves the training curve
  and leaves `eval_lift_sd` on the same scale. The obvious next assumption —
  that reaching `gamma` needs only a new `--gamma` flag — is only half true:
  `clone.collect`'s `gamma` is a *separate* unreachable knob computing the
  demonstrations' value targets, and a flag that moves one and not the other
  makes the inherited critic predict a different quantity from the one PPO
  fits. That is bug 2's shape.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim.train.run` | the only caller of `train` |
| `cr_sim/api/vec.py` | supplies the rollout when `--workers > 0` |
| `cr_sim/train/evaluate.py` | imports `_unflatten_action` only |
| `tests/test_train.py` | holds the seeding to account |

## See

- Source: `cr_sim/train/ppo.py`
- As-built: `docs/training.md`
