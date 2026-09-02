---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/selfplay.py
---

# Self-play opponents and probes

The opponent a run trains against and the two in-run readings taken of it:
`OpponentPool`, `FrozenOpponent`, `PooledOpponent`, `evaluation_probe`,
`ancestor_probe`.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

**Frozen, not live.** Both sides sharing one set of weights makes the opponent
change underneath the rollout that is scoring it, and the advantage estimates
then measure a moving target (`cr_sim/train/selfplay.py:12-15`).

**A pool, not a snapshot.** A single frozen opponent lets the learner cycle —
beat last week's strategy, forget the one before, and go round in circles while
the return says nothing is wrong. Eviction is therefore from the **middle**,
never the ends: the oldest member is the benchmark the ancestor probe is
measured against and the newest is the only one near the learner's strength, so
oldest-first eviction turns the pool back into the sliding window it exists to
avoid (`:381-392`).

**A temperature below 1.0, and this is not a tuning knob.** An early policy has
entropy near the uniform maximum, so an opponent sampling from it at 1.0 is
*still nearly random* — which leaves the critic nothing to fit and the
advantages as noise, the thing self-play was supposed to fix (`:264-273`).

## Shape

- `FrozenOpponent` `:212`. `opponent_name` is in `__slots__` `:220-221` —
  without it the class had no `__dict__` either, so every frozen opponent
  reported as "unknown" and no row it wrote could pass `check_lift_is_named`
  (`:232-243`). `greedy` is a real flag and not `temperature=1e-3`, because
  1e-3 is *nearly* argmax and "nearly reproducible" is how a reproducibility
  claim quietly becomes false (`:245-254`).
- `OpponentPool` `:337` — `add` `:369` (deep-copied to CPU whatever device the
  learner is on; eight networks on an accelerator exhausted it mid-run and
  surfaced as an optimiser step failing), `sample` `:394`, `oldest` `:399`,
  eviction `:381-392`. `generations` is the age scale `ancestor_age` is
  reported on.
- `PooledOpponent` `:403` — adopts a random ancestor on refresh; `net` is
  ignored beyond the signature, so adding a generation and choosing who plays
  stay separate decisions `:413-425`.
- `evaluation_probe` `:426` — lift against a random control on **one fixed
  seed list**, drawn once at `:440`. This is the default probe.
- `ancestor_probe` `:488` — score against `pool.oldest()`. Returns `{}` rather
  than NaN when there is nothing to measure `:513-521`: `np.mean([])` is NaN,
  `json.dumps` writes a bare NaN token that is not valid JSON, and the page
  drew an empty ladder for a whole run because of it. Emits its own
  `ancestor_*` family, never `eval_opponent`, because `run.py` merges this dict
  into one row with the other probes and a shared field means the last writer
  relabels the others `:559-566`.
- Wiring: `cr_sim/train/run.py:953-961` builds the snapshots and the ancestor probe;
  `:922-932` is `_on_refresh`, which adds this generation to the pool **before**
  the opponents draw from it — the other order leaves the pool a generation
  behind for ever.

## Connected to

- **owns:** the `ancestor_*` metrics family and the opponent every `--opponent
  self` rollout faces.
- **owned-by:** [`run-directory.md`](run-directory.md) — the probes exist to
  put a number on a row.
- **joins:** [`lift.md`](lift.md) (`evaluation_probe` emits `eval_lift_sd` at
  `ddof=0`); [`verdict.md`](verdict.md) (`SCORED_FAMILIES` exists because of
  `ancestor_probe`); [`random-streams.md`](random-streams.md) (R2 and R3);
  [`ladder.md`](ladder.md) (`FrozenOpponent` is how a net `Player` plays).
- **looks-like-but-is-not:** `rotating_probe`. Same emitted keys, same
  meanings, same call signature — a drop-in — but it plays block `n % blocks`
  so a three-reading promotion window spans three disjoint sets of battles
  rather than three readings of one (`cr_sim/train/evaluate.py:555`,
  `:565-572`). Swapping them changes which battles a run promotes on and
  nothing a reader can see in the row except `eval_block`.

## If you change this

- **Hits:** the promotion window. `cr_sim/train/run.py:851-866` averages the last three
  readings of `eval_lift_sd` (or `ladder_elo`) and writes `best.pt` on the
  mean, so any change to what a probe returns changes which weights survive
  the run. Changing the pool's eviction changes `ancestor_age`, and an age is
  the whole scale `ancestor_score` sits on.
- **Does not hit:** the workers. Under `--workers N` the rollout opponent is a
  `FrozenOpponent` rebuilt inside each worker from a state dict over a pipe
  (`cr_sim/api/vec.py:234-242`), **not** the `PooledOpponent` the parent holds
  — those still exist and are refreshed but never act. The obvious next
  assumption, that fixing `PooledOpponent` fixes what a `--workers` run plays
  against, is wrong in both directions: that is R2's asymmetry.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim.train.run` | builds the pool, wires both probes, promotes on their output |
| `cr_sim/api/vec.py` | rebuilds `FrozenOpponent` per worker on `set_opponent` |
| `cr_sim/train/watch.py` | draws the self-play ladder from `ancestor_*` rows |
| `cr_sim/train/ladder.py` | reuses `FrozenOpponent` for net players |

## See

- Source: `cr_sim/train/selfplay.py`
