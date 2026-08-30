---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/targeting.py
---

# Targeting

Who a unit decides to attack. Most of Clash Royale's texture lives here rather
than in damage numbers.

## Why this shape

Three rules, each easy to get subtly wrong.

**Range is hitbox-to-hitbox, not centre-to-centre.** A unit reaches a target
when the *gap between hitboxes* closes to its `Range`, so a Giant with a 0.75
tile radius can be hit from further away than a Skeleton with 0.5. Comparing
centres makes every large unit harder to reach than it is.

**Targets are sticky.** A unit does not re-choose each tick; it holds its
target until that target dies or leaves `Range +
LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET` — a grace band that stops flickering
between two equidistant enemies. `RetargetEachTick` opts specific units out.

**Building-targeting troops are not "preferring" buildings — they cannot see
troops at all.** A Giant with a Musketeer in its face has no target other than
the tower, which is why it never stops walking. `target_only_troops` is the
mirror image and Ram Rider's rider is its only standard-deck user: the ram
underneath charges the tower, the rider only ever bolas troops.

Two performance choices are correctness arguments in disguise:

- `within_gap` answers every yes/no range question in squared space with no
  `isqrt`, and the docstring carries the integer proof that
  `isqrt(d2) <= k` is exactly `d2 < (k+1)^2`. Exact, integer, identical on
  every machine.
- `acquire_target` takes a **strict minimum over `(gap, id)`**, and ids are
  unique, so no two candidates can tie and there is exactly one minimum
  whatever order they arrive in. That is the licence for `Battle` to feed it
  `_index.in_reach` — the cheapest enumeration the spatial index can produce —
  instead of a positionally ordered list. Take id uniqueness away
  ([entity-ids.md](entity-ids.md)) and candidate order becomes load-bearing.

## Shape

- `gap_between` / `within_gap` — the two range primitives; only *ranking* needs
  the gap as a number.
- `can_target(spec, attacker, target)` — a **hard filter**, not a preference:
  dead, same team, `UNTARGETABLE_KINDS` (projectiles and area effects — leaving
  them targetable let a Knight kill a 1-HP Poison cloud and cut the spell
  short), not `is_acquirable`, air/ground, buildings-only, troops-only.
- `in_attack_range` — `minimum_range` is the only case that needs the number.
- `acquire_target(..., sight_bonus_for_towers=)` — nearest-first by gap, id as
  tiebreak.
- `should_keep_target(..., range_extension=)` — the stickiness rule.
- `STRUCTURE_KINDS`, `UNTARGETABLE_KINDS`.
- `in_sight_range` (`cr_sim/engine/targeting.py:143`) is a **ghost**: in
  `__all__`, called nowhere; the logic is inlined at
  `cr_sim/engine/targeting.py:180-190` and `cr_sim/engine/battle.py:1119`.
  `nearest_structure` (`cr_sim/engine/targeting.py:229`) is a ghost with zero
  references anywhere.

Citations: `cr_sim/engine/targeting.py:55`, `:72`, `:96`, `:131`, `:147`,
`:160-164`, `:206`, `:49-52`, `:1-25`; `cr_sim/engine/battle.py:1113-1125`
(the sight-reach call), `:403-407` (the two globals).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `STRUCTURE_KINDS`, `UNTARGETABLE_KINDS`.
- **owned-by:** [../build/unit-spec.md](../build/unit-spec.md) — every rule
  reads a spec flag, never an entity flag.
- **joins:** [pathing.md](pathing.md) (`SpatialIndex.in_reach` supplies the
  candidates), [entity.md](entity.md) (`is_acquirable` vs `is_targetable`),
  [attack-state.md](attack-state.md) (a target is the input to the attack
  cycle), [entity-ids.md](entity-ids.md).
- **looks-like-but-is-not:** `Battle._pick_objective`
  (`cr_sim/engine/battle.py:2776`), which is a *movement* goal picked nearest-by-x to
  keep a unit in its lane — a different question from target selection, with a
  different tiebreak.

## If you change this

- **Hits:** every simulated duel and therefore the interaction gate
  ([../build/validation-gates.md](../build/validation-gates.md)) — targeting is
  the layer arithmetic cannot see, so the gate's `computed` vs `simulated`
  disagreements are mostly this file; and every stored hash stream, since
  `target_id` is folded ([state-hash.md](state-hash.md)).
- **Does not hit:** what a unit walks toward when it has no target. That is
  `_nearest_enemy_tower` plus [pathing.md](pathing.md), and a
  building-targeter walking past a Musketeer is *correct* here — the wrong next
  file to open is `pathgrid.py`. Nor does it hit splash: area damage tests
  `is_targetable`, not `can_target`, so an invisible or building-only rule
  never applies to it.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/data/interactions.py` | reads — `simulate_duel` is the observable |
| `tests/test_combat.py`, `test_status_effects.py` | pin |
| nothing outside `cr_sim/engine/` calls these directly | — |

## See

- Source: `cr_sim/engine/targeting.py`
