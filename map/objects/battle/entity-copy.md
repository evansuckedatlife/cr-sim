---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/entity.py
---

# Entity copy paths

`Entity.__deepcopy__` and `Entity.clone` — two flat copies with **different
coverage and different universes**, plus `_slots_for`, the MRO walk both rest
on. Split off [`entity.md`](entity.md) because a single "if you change this"
bullet cannot state two different answers, and stating one for both is how a
mutable slot gets silently aliased.

## Why this shape

Both are flat because entities refer to each other by `target_id` and never by
object reference, so a copied battle needs no reference fixing
([`entity.md`](entity.md)). Both walk `_slots_for(type(self))` rather than
`Entity.__slots__`, because `__slots__` is per class and not inherited:
constructing an `Entity` and iterating the base tuple silently dropped whatever
a `Projectile` or `AreaEffect` adds (`cr_sim/engine/entity.py:115-131`,
`:266-269`).

`__deepcopy__` exists for speed, and its docstring says what it is beating: the
generic machinery reconstructs every entity through `__reduce_ex__`, which for a
`__slots__` class pickles each slot into a state dict and then deep-copies *that
dict*, recursing for every plain `int`. On a mid-match clone that was the
largest single cost (`:276-287`).

## Shape

**`Entity.__deepcopy__` — `cr_sim/engine/entity.py:275-307`. Live, and the only
one that runs.** Slot walk, then `buffs` (`:296-297`), then `struck`
(`:304-306`). `struck` belongs to `RollingProjectile` and `AreaEffect`, which
mutate a set of already-hit ids in place; aliasing it would let a branch's hits
mark the origin's projectile as having struck someone. `spec` / `pspec` /
`aspec` stay aliased on purpose — immutable and already shared through
`Battle._SHARED` ([`battle-clone.md`](battle-clone.md)).

**`Entity.clone` — `cr_sim/engine/entity.py:256-273`. Ghost: zero callers.** It
copies every slot and then `buffs` (`:271-272`) and returns. **There is no
`struck` branch.** Its docstring scopes it to "troops and buildings (the Clone
spell's targets)" (`:288-291`, in `__deepcopy__`'s docstring, explaining why
`clone` needs no subclass coverage) — but **the code wins: nothing calls it.**
The Clone spell does not: `_handle_clone` (`cr_sim/engine/actions.py:1037`)
reaches `Battle._clone_entity` (`cr_sim/engine/battle.py:2579-2627`), which
constructs a **fresh** `Entity(...)` at `:2610` with an explicit field list
rather than copying one. Recorded in
[`../../_meta/overrides.md`](../../_meta/overrides.md).

**The asymmetry, stated once:**

| | walks all slots | copies `buffs` | copies `struck` | callers |
|---|---|---|---|---|
| `__deepcopy__` | yes | yes | **yes** (`:304-306`) | `Battle.clone` (`cr_sim/engine/battle.py:689`), on every search branch |
| `clone` | yes | yes | **no** | **none** |

Citations: `cr_sim/engine/entity.py:115-131` (`_slots_for` and its cache),
`:256-273` (`clone`), `:271-272`, `:275-307` (`__deepcopy__`), `:288-291`,
`:296-297`, `:304-306`; `cr_sim/engine/battle.py:689-740` (`Battle.clone`),
`:2610` (the Clone spell's own construction);
`cr_sim/engine/buffs.py:478` (`BuffState.clone`).
Verified 2026-08-30 against `main` @ `dc47f51`; caller search over `cr_sim/`,
`scripts/` and `tests/`.

## Connected to

- **owns:** `_slots_for` and `_slots_cache`.
- **owned-by:** [`entity.md`](entity.md).
- **joins:** [`battle-clone.md`](battle-clone.md) (`Battle.clone` seeds a
  deepcopy memo with the shared objects rather than enumerating the mutable
  ones — the enumeration is the version that rots),
  [`lookahead.md`](lookahead.md) and
  [`../measurement/search-bot.md`](../measurement/search-bot.md) (one branch per
  candidate per decision), [`state-hash.md`](state-hash.md),
  [`buff-percent.md`](buff-percent.md) (`BuffState.clone`, which both paths
  call).
- **looks-like-but-is-not:** `copy.copy`. Neither path is a shallow copy in
  Python's sense — both construct through `cls.__new__` and assign slots, so
  neither runs `__init__` and neither is reachable through `copy.copy`. And
  `Entity.clone` is not the Clone spell.

## If you change this

- **Hits:** **add a mutable slot and `__deepcopy__` is the one that must
  change.** It names `struck` by hand; it does not detect containers, so a new
  set or list is aliased across every search branch and a branch's writes reach
  the battle it came from. That is a desync `state_hash` will not catch until it
  already diverged ([`state-hash.md`](state-hash.md)). `Battle.clone`'s memo
  covers only `_SHARED`, `_HISTORIES` and the interpreter caches
  (`cr_sim/engine/battle.py:704-738`) — a new slot is not on any of those lists.
- **Does not hit:** `Entity.clone`, which nothing calls, and `spec` / `pspec` /
  `aspec`. Those are shared deliberately; the obvious next move on finding an
  aliased attribute — deep-copying the spec too — undoes `Battle._SHARED` and
  makes every branch allocate the whole build.

## Surfaces

| Surface | Role |
|---|---|
| `Battle.clone` → `SearchBot` (`cr_sim/train/scripted.py:471`) | the hot caller: one copy per candidate per decision |
| `cr_sim/engine/lookahead.py:122` | the other caller |
| `tests/test_collision.py:402`, `tests/test_lookahead.py` | pin that a branch and its origin stay independent |
| `Entity.clone` | **no surface** |

## See

- Source: `cr_sim/engine/entity.py`
