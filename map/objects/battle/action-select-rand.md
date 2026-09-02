---
type: object
cluster: battle
universe: deliberate ghost
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionSelect with `Condition = "rand(n)"`

The *other* spelling of `ActionSelect`: pick a sub-action at random rather than
by condition. Its own card because a noun with two universes needs two cards —
[`action-select.md`](action-select.md) is `live`, this spelling is a
**deliberate ghost**, and a paragraph inside a `universe: live` card is not
queryable as the thing it is.

## Why this shape

Always taking branch zero would turn the Spell Cauldron into a Lightning
dispenser. The interpreter is given **no random stream on purpose** — every
draw in this repo is owned by a named generator derived arithmetically from
something a reader can see ([`rng.md`](rng.md),
[`../measurement/random-streams.md`](../measurement/random-streams.md)) — and
handing the ACTION interpreter one would put an unowned draw inside the tick
loop, where [`state-hash.md`](state-hash.md) makes it a replay desync rather
than a wrong number.

The content behind it is event-only: the Spell Cauldron, Blackout, the gift
spawners. Nothing a standard deck can play reaches it. So the branch is
**recorded as a gap** instead of resolved, and the gap is the tripwire: a
non-zero `<select:rand(n)>` count means a card reached it and was declined,
which is a different fact from "the node is broken".

## Shape

- Declined, with the reason, at `cr_sim/engine/actions.py:73-78`.
- Recorded at `cr_sim/engine/actions.py:902-907`: where a node has a `Condition`
  and no `PerActionConditions`, `interp.unsupported["<select:" + chooser + ">"]`
  is incremented and the handler returns without scheduling anything.
- Of the 20 `ActionSelect` nodes in this build, **7** carry
  `Condition = "rand(n)"` ([`action-select.md`](action-select.md), where that
  count is derived).
- `_passes` never sees it: `ActionSelect` is the one class type for which the
  gate reads only `ExecuteIfTrue` (`cr_sim/engine/actions.py:641-656`).

Citations: `cr_sim/engine/actions.py:73-78`, `:641-656`, `:871-895` (the
handler docstring), `:902-907`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** the `<select:rand(n)>` key in `ActionInterpreter.unsupported`.
- **owned-by:** [`action-select.md`](action-select.md).
- **joins:** [`rng.md`](rng.md), [`action-interpreter.md`](action-interpreter.md),
  [`action-taunt.md`](action-taunt.md),
  [`action-ground-to-air.md`](action-ground-to-air.md),
  [`action-change-data-projectile.md`](action-change-data-projectile.md) — the
  other three deliberate ghosts.
- **looks-like-but-is-not:** an unimplemented class type. `ActionSelect` **is**
  in `_HANDLERS` (`cr_sim/engine/actions.py:1554`); this is one spelling of a
  live node, which is why the gap key is bracketed rather than a class name.

## If you change this

- **Hits:** implementing it means giving the interpreter a random stream, which
  is a change to [`rng.md`](rng.md) and therefore to
  [`state-hash.md`](state-hash.md) and every replay. Seven nodes move out of the
  gap list and into live behaviour — a **mechanics change**, not a fix, and one
  no test in `tests/` currently expects.
- **Does not hit:** the coverage total or any card a standard deck plays.
  The obvious wrong reading — that a non-zero `<select:rand(n)>` count is a bug
  — is the thing this card exists to prevent.

## Surfaces

| Surface | Role |
|---|---|
| `ActionInterpreter.unsupported` | the only place the decline is visible |
| a future taunt/event implementer | the reader this card is written for |
| `tests/` | none — nothing asserts the count |

## See

- Source: `cr_sim/engine/actions.py`
