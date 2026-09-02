---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/entity.py
---

# Entity

Anything that occupies a position and can be interacted with: troops,
buildings, towers, projectiles and area effects. What the tick loop iterates.

## Why this shape

Two decisions here are hard to retrofit and everything else depends on them.

**`__slots__` everywhere.** A three-minute battle is 10 800 ticks and can hold
dozens of entities, so this is millions of attribute reads; slots remove the
per-instance dict and cut both memory and lookup cost. The cost is that
subclasses each need their own `__slots__`, which is why `_slots_for` collects
the whole MRO once per class and caches it — iterating `Entity.__slots__` and
constructing an `Entity` silently dropped whatever a `Projectile` added.

**Entities are never removed from the list mid-tick.** Death sets a flag; the
sweep is a defined phase (`resolve_deaths`, 18th of 20). Iteration order stays
stable and every phase within a tick sees the same population. Mutating the
list while phases iterate it is the classic source of order-dependent,
irreproducible bugs — and it is the precondition
[state-hash.md](state-hash.md) rests on.

`is_targetable` and `is_acquirable` are **deliberately not the same test**. An
invisible Royal Ghost cannot be *chosen* as a target, but a Fireball dropped on
its tile still burns it. Splash and area effects test `is_targetable`; only
target selection tests `is_acquirable`. Collapsing them makes invisibility a
blanket immunity.

`spec` is shared, never copied — it is immutable and lives in
`Battle._SHARED` ([battle-clone.md](battle-clone.md)). `buffs` is created
lazily because most entities never carry one.

## Shape

- `Entity.__slots__` — **21**: id, kind, team, spec, x, y, hitpoints,
  max_hitpoints, shield, state, state_ticks, spawn_tick, deploy_ticks_left,
  target_id, collision_radius, mass, flying, dead, lifetime_left, buffs,
  is_clone.
- `Team` — `BLUE = 0` defends the low-y end; `.opponent`.
- `EntityKind` — TROOP, BUILDING, TOWER, PROJECTILE, AREA_EFFECT.
- `EntityState` — DEPLOYING, IDLE, MOVING, ATTACKING, DYING, DEAD.
  `DEPLOYING` is a real state, not a formality: the window is what makes a
  surprise placement punishable.
- `is_clone` exists because the build's `CLONE_CLONED_UNITS` global is False —
  a second Clone must not double the first one's output.
- Entities refer to each other by `target_id`, never by object reference. That
  is what makes both copy paths flat and reference-fixing unnecessary.

Citations: `cr_sim/engine/entity.py:134-159`, `:36`, `:47`, `:55`, `:212-252`,
`:115-131`, `:256`, `:275`, `:348`; `cr_sim/engine/battle.py:243-263` (the
sweep's position in the order).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `Team`, `EntityKind`, `EntityState`, `_slots_for`,
  [entity-ids.md](entity-ids.md).
- **owned-by:** [../build/unit-spec.md](../build/unit-spec.md) — an entity is a
  position plus a pointer to a spec.
- **joins:** [state-hash.md](state-hash.md) (11 of the 21 slots are hashed),
  [targeting.md](targeting.md), [attack-state.md](attack-state.md),
  [buff-percent.md](buff-percent.md) (`buffs`),
  [battle-clone.md](battle-clone.md).
- **looks-like-but-is-not:** `Projectile`, `RollingProjectile`, `AreaEffect` —
  subclasses with their own `__slots__` and their own `pspec`/`aspec`. Code
  that reads `entity.spec` and assumes a `UnitSpec` will find `None` on them;
  `Battle._capture_frame` falls back through `pspec` for exactly that reason.

## If you change this

- **Hits:** `state_hash` if the added slot is folded in, and **the two copy
  paths, which do not have the same coverage** —
  [`entity-copy.md`](entity-copy.md) owns that waterfall and is where a new
  mutable slot has to be checked against both. A single sentence about "both
  copy paths" is wrong for one of them, which is why it is a card and not a
  bullet.
- **Does not hit:** targeting rules. `can_target` reads the *spec*, not the
  entity, for air/ground and building-only — adding an entity flag does not
  make anything see it. The obvious wrong place to add "this unit is
  untargetable" is a slot here; `UNTARGETABLE_KINDS` and `is_acquirable` are
  where that lives ([targeting.md](targeting.md)).

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/api/encoding.py` | reads — entities become observation channels |
| `Battle._capture_frame` → the HTML viewer | reads 10 fields per entity |
| `cr_sim/api/reward.py` | reads — unit value, deaths |
| `tests/test_engine_core.py`, `test_collision.py` | pin the lifecycle |

## See

- Source: `cr_sim/engine/entity.py`
