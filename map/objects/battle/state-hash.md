---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/replay.py
---

# state_hash

The whole simulation state folded into one integer per tick. How the
determinism claim gets *enforced* rather than merely intended.

## Why this shape

Two runs that agree on every tick's hash are identical simulations; two that
diverge report **the exact tick where they first differed**, which turns "the
replay desyncs" from a hunt into a single-step diff. `compare_hashes` returns
that tick rather than a boolean for exactly this reason.

**Entities are hashed in list order, not sorted.** That is why entities are
never removed from the list mid-tick ([entity.md](entity.md)): list order is
spawn order and never mutated, so the digest is stable without paying for a
sort. **Entity-list stability is therefore a correctness precondition of the
hash**, not an implementation detail — any change that reorders, compacts or
filters `Battle.entities` inside a tick invalidates every stored stream.

**Only fields that can affect the future are folded**: id, team, kind, state,
dead, x, y, hitpoints, shield, `deploy_ticks_left`, `target_id`. Anything
cosmetic or derived is excluded on purpose, so adding a debug counter or a
render hint cannot invalidate a stored replay. `Battle.frames` is outside for
that reason.

Note what this means in practice: `entity.id` is folded, so an id shift with no
physical difference *is* a divergence ([entity-ids.md](entity-ids.md)). And
elixir is not an entity, so `Battle.hash` passes both bars in as `extra`.

## Shape

- `state_hash(tick, entities, extra=()) -> int` — blake2b, 8-byte digest, little
  endian; tick first, then eleven fields per entity, then the extras.
- `Battle.hash()` — `state_hash(tick, entities, extra=(blue elixir, red
  elixir))`, in fixed-point thousandths.
- `compare_hashes(left, right) -> int | None` — first diverging tick, or the
  shorter length when one stream is a prefix of the other.
- `DivergenceError` — defined and exported; **raised nowhere in the repo.**

Citations: `cr_sim/replay.py:38`, `:41-45`, `:47-66`, `:165`, `:34`;
`cr_sim/engine/battle.py:841`; `cr_sim/engine/entity.py:13-17` (why the list is
never mutated mid-tick).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `compare_hashes`, `DivergenceError`.
- **owned-by:** [entity.md](entity.md) — the field list is a promise about
  `Entity`'s slots.
- **joins:** [battle.md](battle.md) (`Battle.hash`),
  [entity-ids.md](entity-ids.md), [replay.md](replay.md),
  [../build/tick-clock.md](../build/tick-clock.md) — a 20 TPS run and a 60 TPS
  run produce different streams by construction; comparing them is meaningless.
- **looks-like-but-is-not:** a checkpoint hash or a config hash. Nothing in
  `cr_sim/train/` hashes a battle for provenance; the only training-side reader
  is `info["hash"]` (`cr_sim/api/env.py:236`).

## If you change this

- **Hits:** every stored hash stream at once — the four determinism tests that
  run a battle twice and assert `compare_hashes(...) is None`
  (`tests/test_arena_and_battle.py:468`, `tests/test_collision.py:286`,
  `tests/test_combat.py:413`, `tests/test_projectiles.py:430`), and `info["hash"]`, which
  `cr_sim/train/scripted.py` uses to notice a proposer changed the match.
  Adding a field to the fold is a one-way door for any replay recorded before
  it.
- **Does not hit:** whether the simulation is *correct*. A matching hash proves
  two runs agree, not that either is right — that is the interaction gate's job
  ([../build/validation-gates.md](../build/validation-gates.md)). The obvious
  wrong inference is that a green determinism test means a mechanics change was
  safe: both runs changed identically.

## Surfaces

| Surface | Role |
|---|---|
| `cr-sim battle` prints `state hash:` (`cr_sim/cli.py:446`) | reads |
| `cr_sim/api/env.py:236` → `info["hash"]` | reads — every training step carries one |
| `tests/test_engine_core.py:376-395` | pins the fold and the divergence report |
| `Replay.hashes` | stores, when recording ([replay.md](replay.md)) |

## See

- Source: `cr_sim/replay.py`
