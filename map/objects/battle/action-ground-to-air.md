---
type: object
cluster: battle
universe: deliberate ghost
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionGroundToAir

The inverse of `ActionAirToGround`: lift a grounded entity back into the air.
**Not implemented, on purpose** — its only user is a hero form whose ability
graph is not wired up at all.

## Why this shape

`ActionAirToGround` **is** implemented (`cr_sim/engine/actions.py:1272`,
registered at `:1565`), because Vines and the Hunter evolution's net pull flyers
down and both are cards a deck can play. The inverse has exactly one user in the
build, `WizardHero`, and hero ability graphs are not wired up — so implementing
it would add a code path with no reachable caller and no way to check it.

Implementing the inverse in isolation is also the wrong shape: the grounding
this would undo is tracked in `Battle._grounded`, a dict of entity id to
remaining ticks (`cr_sim/engine/battle.py:399-401`). A lift that does not
reconcile with that timer produces an entity that is airborne and still counted
as grounded.

## Shape

- Declined, with the reason, at `cr_sim/engine/actions.py:79-82`.
- No entry in `_HANDLERS`; the implemented inverse is
  `"ActionAirToGround": _handle_air_to_ground` (`cr_sim/engine/actions.py:1565`).
- The state it would have to unwind: `Battle._grounded`
  (`cr_sim/engine/battle.py:399-401`).

Citations: `cr_sim/engine/actions.py:56-58`, `:79-82`, `:1272`, `:1565`;
`cr_sim/engine/battle.py:399-401`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing.
- **owned-by:** [`action-interpreter.md`](action-interpreter.md).
- **joins:** [`targeting.md`](targeting.md) (air/ground admissibility),
  [`action-taunt.md`](action-taunt.md),
  [`action-select-rand.md`](action-select-rand.md),
  [`action-change-data-projectile.md`](action-change-data-projectile.md).
- **looks-like-but-is-not:** `ActionAirToGround`, which is live and which Vines
  and the Hunter evolution both reach.

## If you change this

- **Hits:** `Battle._grounded` and the flying flag on
  [`entity.md`](entity.md) — both, together, or the entity's two notions of
  where it is disagree.
- **Does not hit:** Vines or the Hunter evolution. Those use the *forward*
  direction, which already works; the obvious wrong assumption, that a missing
  `ActionGroundToAir` is why something lands and never takes off again, is a
  question about the `_grounded` timer instead.

## Surfaces

| Surface | Role |
|---|---|
| `ActionInterpreter.unsupported` | records the class type by name |
| `WizardHero` | the build's only user, itself unwired |

## See

- Source: `cr_sim/engine/actions.py`
