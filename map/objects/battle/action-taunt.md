---
type: object
cluster: battle
universe: deliberate ghost
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionTaunt

Clears a target lock. **Not implemented, on purpose** — left absent rather than
stubbed, so that adding a taunt mechanic later trips the coverage gate instead
of inheriting a silent no-op.

## Why this shape

The build uses it to *clear* a lock: Goblin Demolisher's transformation runs one
so its kamikaze form can pick a building. This engine has no taunt and no lock,
so there is nothing to clear. A stub that returned immediately would be
indistinguishable from a correct implementation for as long as nothing needed
it, and the moment something did, it would be a no-op nobody could find.

Absent, the class type lands in `ActionInterpreter.unsupported` as a **name**,
which is the whole design of that counter: a card that quietly does nothing
shows up as a name rather than as a mystery
([`action-interpreter.md`](action-interpreter.md)).

**Implementing this without reading its reason removes a tripwire.**

## Shape

- Declined, with the reason, at `cr_sim/engine/actions.py:59-64`.
- No entry in `_HANDLERS` (`cr_sim/engine/actions.py:1548-1583`), which is what
  puts the class type in `unsupported` rather than running anything.
- Nothing in `cr_sim/engine/` implements a target lock for it to clear —
  `can_target` / `is_acquirable` are the whole of target admissibility
  ([`targeting.md`](targeting.md)).

Citations: `cr_sim/engine/actions.py:56-58` (the four gaps are deliberate),
`:59-64`, `:1548-1571`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing.
- **owned-by:** [`action-interpreter.md`](action-interpreter.md).
- **joins:** [`targeting.md`](targeting.md) (where a lock would have to live),
  [`action-select-rand.md`](action-select-rand.md),
  [`action-ground-to-air.md`](action-ground-to-air.md),
  [`action-change-data-projectile.md`](action-change-data-projectile.md).
- **looks-like-but-is-not:** a missing feature. It is a declined one, and the
  decline is load-bearing.

## If you change this

- **Hits:** [`targeting.md`](targeting.md) first — a taunt needs a lock, and
  this engine has none — then `ActionInterpreter.unsupported`'s totals and every
  coverage assertion that reads them.
- **Does not hit:** Goblin Demolisher, which works. Its transformation is
  `ActionChangeGameObjectData`'s character form and is implemented
  (`cr_sim/engine/actions.py:1380-1414`); the taunt beside it does nothing here
  because there is nothing for it to undo.

## Surfaces

| Surface | Role |
|---|---|
| `ActionInterpreter.unsupported` | records the class type by name |
| a future taunt implementer | the reader this card is written for |

## See

- Source: `cr_sim/engine/actions.py`
