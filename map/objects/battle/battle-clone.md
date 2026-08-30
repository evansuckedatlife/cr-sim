---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/battle.py
---

# Battle.clone

An independent battle continuing from this position, produced by a
memo-seeded `deepcopy`. What makes "ask what happens next" cheap enough to be a
reward term.

## Why this shape

A naive `deepcopy` of a mid-match `Battle` cost **five times more than
simulating an entire match**, because the object graph reaches the whole card
database. So the copy is inverted: instead of enumerating what to copy, it
seeds the memo with what may be **shared**, and everything else is copied by
default.

That direction is the load-bearing decision. An enumeration of the mutable
slots is the version that rots — a slot added later would silently stay shared
between a battle and its branches, and the resulting bug would look like
nondeterminism rather than like a missing copy. As written, **a new slot
defaults to being copied**, which is the safe direction: the cost of forgetting
is a slower clone, not a corrupted one.

Two further exclusions rest on one invariant — *nothing rereads or mutates what
is already in an append-only history*:

- `_HISTORIES` (`graveyard`, `damage_log`, `frames`) are set aside and the
  branch gets fresh containers. On a mid-match position these three were more
  than half the cost of a copy.
- The corpses go into the memo **as themselves**, because `_by_id_map` keeps
  every entity ever registered reachable by id, dead included — that is what
  makes a stale `target_id` resolve to a corpse rather than to nothing — so
  deepcopy was reaching the whole graveyard through the map and rebuilding a
  couple of hundred entities per clone, leaving the branch with *two* objects
  for one dead unit.

`Entity.__deepcopy__` is the matching fast path: slot-by-slot for every
concrete subclass, with `buffs` and `struck` given real copies because they are
the only mutable containers, and `spec` deliberately left aliased.

## Shape

- `_SHARED` — 12 names: `config`, `clock`, `data`, `levels`, `registry`,
  `arena`, `timeline`, and the five name-keyed spec caches. All immutable or
  caches of immutables.
- `_HISTORIES` — 3 names, restored on the original in a `finally`.
- Also memoised: `actions._cache` and `actions.unsupported`.
- `clone.frames = []` unconditionally — a branch has no viewer.
- `Entity.clone()` (`cr_sim/engine/entity.py:256`) is a *different* thing: the Clone
  spell's flat copy, troops and buildings only.

Citations: `cr_sim/engine/battle.py:680`, `:687`, `:689`, `:707-750`;
`cr_sim/engine/entity.py:275-305`, `:256`; `cr_sim/engine/pathgrid.py:109`
(`PathGrid.__deepcopy__`).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `_SHARED`, `_HISTORIES`.
- **owned-by:** [battle.md](battle.md).
- **joins:** [lookahead.md](lookahead.md) (the only production caller,
  `cr_sim/engine/lookahead.py:122`), [entity.md](entity.md),
  [entity-ids.md](entity-ids.md) — a clone spawns, and spawning burns ids the
  live battle was going to use.
- **looks-like-but-is-not:** `Entity.clone` (the spell) and
  `torch.Tensor.clone` (`cr_sim/api/vec.py:413`). Three unrelated `clone`s.

## If you change this

- **Hits:** [lookahead.md](lookahead.md) and every reward that uses a
  projection; `cr_sim/train/scripted.py:471`, the second caller; and the
  branch's isolation — a mutable object added to `_SHARED` makes a branch
  write into the battle it came from, which surfaces as an irreproducible
  match, not as a copy error.
- **Does not hit:** the entity id counter. `clone` does not touch it; ids are
  burned by what the branch *spawns*, and handing them back is
  [entity-ids.md](entity-ids.md)'s job, done by `project`, not here. The
  obvious wrong fix for "a projection changed the match" is to add id handling
  to `clone`; the cursor already lives at the call site.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/engine/lookahead.py:122` | reads |
| `cr_sim/train/scripted.py:471` | reads (working-tree line — see `../../CONTEXT.md`) |
| `scripts/bench_engine.py:194` | reads |
| `tests/test_lookahead.py`, `tests/test_collision.py:402` | pin isolation and the fast path |

## See

- Source: `cr_sim/engine/battle.py`
