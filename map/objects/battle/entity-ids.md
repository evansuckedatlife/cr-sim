---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/entity.py
---

# Entity ids

A monotonic, never-reused integer, and the pair of functions that let
speculative work hand back the ids it burned. **The** deterministic tiebreak in
this engine.

## Why this shape

Ids are assigned in spawn order and never reused, which gives every phase a
stable tiebreak when two entities are otherwise equivalent — which target to
pick, which of two simultaneous deaths resolves first. `acquire_target` takes a
strict minimum over `(gap, id)` keys and ids are unique, so **no two candidates
can tie** — which is what lets the caller feed it the cheapest enumeration the
spatial index can produce rather than a positionally ordered one
(`cr_sim/engine/targeting.py:160-164`, `cr_sim/engine/battle.py:1113-1125`). Take the
uniqueness away and candidate order becomes load-bearing, silently.

The counter is **module-global**. That buys the tiebreak and costs three
things, all real:

1. Two concurrent battles in one process share it — `reset_entity_ids()` runs
   once per `Battle.__init__` (`cr_sim/engine/battle.py:349`), so constructing a
   second battle resets the counter under the first one.
2. A branch played forward burns ids the live battle was going to use, so
   *asking* what happens next changes what happens next. `entity_id_cursor()` /
   `restore_entity_ids()` are the undo, and `project` takes its cursor **after**
   the card is already on the board — that asymmetry is what leaked ids before.
3. `state_hash` folds `entity.id` ([state-hash.md](state-hash.md)), so an id
   shift is a hash divergence with no physical difference.
   `cr_sim/train/scripted.py:437-449` says exactly that: physics survived, the
   hash did not.

`restore_entity_ids` is only sound when everything allocated since is
unreachable — reviving an id a live entity still holds collides in
`Battle._by_id_map`.

## Shape

- `next_entity_id()` — `global _next_id; _next_id += 1`.
- `entity_id_cursor()` / `restore_entity_ids(cursor)` — the undo pair.
- `reset_entity_ids()` — called once per battle construction.
- Three call sites of the undo pair: `cr_sim/engine/lookahead.py:139`/`:152`,
  `cr_sim/train/scripted.py:467`/`:470`/`:486`, and `tests/test_lookahead.py`.

Citations: `cr_sim/engine/entity.py:74`, `:86`, `:99`, `:109`;
`cr_sim/engine/battle.py:349`; `cr_sim/engine/lookahead.py:135-152`;
`cr_sim/engine/targeting.py:196-201`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing; it is three functions and a module global.
- **owned-by:** [entity.md](entity.md) — every `Entity.__init__` calls it.
- **joins:** [state-hash.md](state-hash.md) (ids are hashed),
  [targeting.md](targeting.md) (the tiebreak),
  [lookahead.md](lookahead.md) (the undo),
  [battle-clone.md](battle-clone.md), [battle.md](battle.md)
  (`_by_id_map`, `_occupancy_signature` and every id-keyed cache).
- **looks-like-but-is-not:** a seed. Ids are not random and are not derived
  from [rng.md](rng.md); a different seed does not change them, and restoring
  the counter does not restore any RNG state.

## If you change this

- **Hits:** every stored hash stream ([state-hash.md](state-hash.md)) and
  therefore every replay comparison; `acquire_target`'s tiebreak and
  `_nearest_enemy_tower`'s (`cr_sim/engine/battle.py:2789`); every id-keyed cache on
  `Battle` (`_attacks`, `_routes`, `_charge`, `_spawn_timers`,
  `_spawn_children`, `_hit_counts`, `_grounded`, `_counters`, `_last_attack`,
  `_by_id_map`) — reuse an id and a dead unit's attack state becomes a live
  one's.
- **Does not hit:** reproducibility of a *training run*. The five unowned
  random streams of bug 5 were `torch`'s global generator under `--workers`,
  numpy's legacy `RandomState` in the PPO shuffle, `evaluation_probe`,
  `ancestor_probe` and `evaluate_paired`; none of them is an entity id, and
  none is fixed by anything in this file. The obvious wrong move after a
  non-reproducible run is to go looking at the id counter — look at
  [rng.md](rng.md) and then at the measurement cluster.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/engine/lookahead.py` | reads / restores |
| `cr_sim/train/scripted.py` | reads / restores (working-tree lines) |
| `tests/test_lookahead.py:76-109`, `tests/test_collision.py:225` | pin the undo |
| nothing outside `cr_sim/` | ids are never serialised except inside a frame |

## See

- Source: `cr_sim/engine/entity.py`
