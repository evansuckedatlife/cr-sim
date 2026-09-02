---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: map/objects/measurement/random-streams.md
---

# Random stream ownership

Every draw in the measurement path, and who owns it. Not a type in the tree —
this is the cross-cutting card, and the only noun in this map whose entity is
the map itself. Bug 5 lives here.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

An unreproducible yardstick cannot settle whether a change moved anything. The
promotion probe returned three different lifts on identical inputs because the
sampled draw came off torch's global stream; multi-worker training was never
replayable because spawned processes seed themselves from OS entropy. Five
streams were closed in `8fbe4a5`. The point of a card rather than five index
lines is that **ownership is a property of the graph, not of a function**: a
stream is owned when nothing between the seed and the draw can advance it, and
that is a fact about call order across files.

Two derivation rules the whole tree follows, and both are load-bearing.
Arithmetic, **never `hash()`** — Python salts string hashing per process, so a
hash-derived stream is reproducible inside one run and not between two
(`cr_sim/train/scripted.py:68-71`, `cr_sim/train/proposal.py:71-75`,
`cr_sim/train/ladder.py:504-506`). And keyed on the **thing being played**, not
on a position in a list.

## Shape

Closed in `8fbe4a5` — five:

- worker self-play opponent: `VecEnvConfig.seed`, made distinct per worker,
  one generator per worker carried across refreshes.
  `cr_sim/api/vec.py:101`, `:192-197`, `:234-242`, `:329`.
- PPO minibatch shuffle: its own `default_rng(config.seed + 1)`, and
  **required** rather than defaulted on `_update`, so no single-line mutation
  can silently replay one permutation. `cr_sim/train/ppo.py:212-222`,
  `:471-483`.
- `evaluation_probe`'s sampled arm: `cr_sim/train/selfplay.py:467`.
- `ancestor_probe`, both sides: `:532-534`.
- `evaluate_paired`, keyed on **which mode this is**, never its index in
  `modes` — the index keying made the sampled arm's stream depend on whether
  the caller also asked for greedy. `cr_sim/train/evaluate.py:457-459`; both
  single-arm callers are live (`scripts/run_ladder.py:380`,
  `scripts/evaluate_vs_expert.py:145`).

Also owned: `ladder._stream_seed` `cr_sim/train/ladder.py:501` (keys on the
mode itself, which is what made `evaluate_paired`'s index keying a slip rather
than a choice); `rotating_probe` `cr_sim/train/evaluate.py:641`; `evaluation_seeds`
`:320`; `battle_stream_seed` `cr_sim/train/scripted.py:52`;
`policy_proposer` `cr_sim/train/proposal.py:144-146`; the clone holdout split
`cr_sim/train/clone.py:592`; and the random control arm's `default_rng(0)`
`cr_sim/train/evaluate.py:176` — a literal no flag reaches, pinned so as not to move the
scale every historical number sits on.

Open, four:

- **R1 — `clone_policy`'s sampled arm has no generator.**
  `scripts/clone_policy.py:316-317` calls `evaluate(..., greedy=False)` with
  no `generator`, so it falls to `Categorical(...).sample()` on torch's global
  stream (`cr_sim/train/evaluate.py:205-206`). `clone()` seeds that stream (`cr_sim/train/clone.py:575`) and
  then advances it once per epoch with an unseeded `randperm` (`:598`). The
  sampled lift in every `runs/clone-*/verdict.json` is therefore a function of
  `--epochs` and `--fraction` as well as of the weights. Every other paired
  evaluator in the tree passes one.
- **R2 — `PooledOpponent` gets no generator.** `clone_policy`'s omission is a
  slip; this one is documented as back-compat (`cr_sim/train/selfplay.py:255-262`). Under
  `--workers 0` the self-play opponent samples off torch's global stream
  (`:329-330`), shared with the learner's own action sampling. Asymmetric with
  `--workers N`, where the worker's `FrozenOpponent` does own one
  (`cr_sim/api/vec.py:234-242`). `PooledOpponent.__init__` never forwards a
  `generator` (`cr_sim/train/selfplay.py:408-411`).
- **R3 — under the default `--opponent self`, every worker resets from
  `default_rng(0)`.** `opponent_seed` is `None` there (`cr_sim/train/run.py:1013`), the
  per-worker `replace` is guarded on it being not-None (`cr_sim/api/vec.py:330-333`),
  and the worker's episode-reset generator is `default_rng(config.opponent_seed
  or 0)` (`:198`), drawn at `:223`. Worker 0 and worker 7 draw the same battle
  sequence. That is the exact hazard the comment at `:321-323` says the design
  guards against for opponents.
- **R4 — `CRSimEnv.reset(seed=None)` seeds from OS entropy.**
  `cr_sim/api/env.py:384-385` and `CRSimSelfPlayEnv` at `:694-695`. Unreached
  from the measurement path today — one omitted keyword away.

## Connected to

- **owns:** nothing in the tree; it is an index over other cards' fields.
- **owned-by:** [`lift.md`](lift.md) — a stream matters here because an
  unreproducible arm makes the lift unreproducible.
- **joins:** [`self-play.md`](self-play.md) (R2, R3),
  [`demonstrations.md`](demonstrations.md) (R1, and the holdout split),
  [`ppo.md`](ppo.md) (the shuffler), [`ladder.md`](ladder.md) (`_stream_seed`),
  [`search-bot.md`](search-bot.md) (`battle_stream_seed`, the proposer).
- **looks-like-but-is-not:** `--seed`. It seeds the run, not the streams:
  `torch.manual_seed(config.seed)` at `cr_sim/train/ppo.py:211` covers the learner's own
  sampling in *this* process and nothing in a spawned one, which is what
  `VecEnvConfig.seed` exists for.

## If you change this

- **Hits:** the number, immediately and silently. Giving R1 or R2 a generator
  changes what those arms play, so every lift measured after the change is on
  a different draw from every lift measured before it — the same class of
  move as re-seeding `_random_opponent`, which is left alone for exactly that
  reason (`cr_sim/train/evaluate.py:259-262`). Closing R3 changes which battles a
  `--workers` run trains on.
- **Does not hit:** the greedy arm, ever. Greedy touches no generator at all
  (`cr_sim/train/evaluate.py:196-197`, `cr_sim/train/selfplay.py:322-323`, `cr_sim/train/proposal.py:133-137`), which
  is why the ladder is greedy and why greedy reproduced bit-identically in the
  noise measurement. The obvious next move — re-running an old greedy verdict
  to check a stream fix — measures nothing, because it was never affected.
  Nor does it hit the engine: battle determinism is `BattleConfig.seed` and
  belongs to `battle`, not here.

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_train.py`, `tests/test_proposal.py` | assert the closed streams stay closed |
| `runs/*/verdict.json` | carry the numbers the open streams affect |
| a human re-running a measurement | the only party that notices a stream moved |

## See

- Source: `cr_sim/api/vec.py`, `cr_sim/train/proposal.py`
