---
type: object
cluster: battle
universe: deliberate ghost
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionChangeGameObjectData with `NewProjectileData`

The projectile spelling of a live node. `NewCharacterData` — swap what a unit
*is* — is implemented; `NewProjectileData` — swap what it *throws* — is
**declined on purpose** and recorded as `<changedata:projectile>`.

## Why this shape

A unit's damage in this engine is resolved **from its projectile at spec-build
time** ([`../build/unit-spec.md`](../build/unit-spec.md)), so swapping the
projectile on a live entity means rebuilding the damage as well. The only two
users are the Executioner and Snowball evolutions, and neither gives a way to
check the result — so guessing at the rebuilt damage would be worse than leaving
the gap visible.

This is the sharpest of the four declines, because the *class type* is
implemented. A coverage counter keyed on class names would report
`ActionChangeGameObjectData` as covered and be right about the character form
and wrong about this one. That is why the gap key is bracketed rather than a
class name ([`action-interpreter.md`](action-interpreter.md)).

## Shape

- Declined, with the reason, at `cr_sim/engine/actions.py:66-72`, and again in
  the handler's own docstring at `:1399-1403`.
- The live half: `_handle_change_data` (`cr_sim/engine/actions.py:1380-1414`),
  registered at `:1567`. Two standard cards turn on it, both at 50% health —
  Cannon Cart's `MovingCannon` → `BrokenCannon`, and Goblin Demolisher →
  `GoblinDemolisher_kamikaze_form`. **Current hitpoints are carried across
  rather than reset**, and the evidence for that reading is in the docstring:
  the Tombstone hero's swap is the only one that chains a
  `Tombstone_hero_ResetHealthValue` afterwards, which would be redundant if the
  swap reset health by itself (`:1393-1397`).
- The decline: `NewCharacterData` absent and `NewProjectileData` present
  increments `interp.unsupported["<changedata:projectile>"]` and returns
  (`:1408-1412`).

Citations: `cr_sim/engine/actions.py:56-58`, `:66-72`, `:1380-1414`,
`:1393-1397`, `:1399-1403`, `:1408-1412`, `:1567`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** the `<changedata:projectile>` key in
  `ActionInterpreter.unsupported`.
- **owned-by:** [`action-interpreter.md`](action-interpreter.md).
- **joins:** [`../build/unit-spec.md`](../build/unit-spec.md) (where damage is
  resolved from a projectile, once), [`attack-state.md`](attack-state.md),
  [`action-taunt.md`](action-taunt.md),
  [`action-select-rand.md`](action-select-rand.md),
  [`action-ground-to-air.md`](action-ground-to-air.md).
- **looks-like-but-is-not:** the character form, which is live and which two
  standard cards depend on. A change to `_handle_change_data` touches both.

## If you change this

- **Hits:** [`../build/unit-spec.md`](../build/unit-spec.md). Implementing it
  means rebuilding a live entity's damage from a new projectile, which is the
  one conversion this engine deliberately does exactly once, at spec-build time
  ([`../../CONTEXT.md`](../../CONTEXT.md), unit conventions). Two evolutions
  move out of the gap list.
- **Does not hit:** Cannon Cart or Goblin Demolisher. Both use the character
  form and both work — the obvious wrong reading of a non-zero
  `<changedata:projectile>` count is that this node is broken.

## Surfaces

| Surface | Role |
|---|---|
| `ActionInterpreter.unsupported` | records the declined spelling |
| `tests/test_evolutions.py` | pins the character form, not this one |
| the Executioner and Snowball evolutions | the two cards that reach it |

## See

- Source: `cr_sim/engine/actions.py`
