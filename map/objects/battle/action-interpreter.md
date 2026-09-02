---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionInterpreter

Runs the build's 828 `ACTION` graphs: a declarative node language for
everything modern Clash Royale does not express as a stat column. The battle
drains a delay queue once per tick.

## Why this shape

Behaviour stopped living in columns years ago. The reworked Graveyard, the
reworked Goblin Hut, champion abilities, evolutions and the King Tower's
activation are all `ACTION` graphs. The graph is data, so an interpreter reads
it — a hundred hand-coded special cases was the alternative.

Four things had to be worked out from the files before any of it could run, and
each is a trap for a change. **The module docstring says "Three" and then bolds
four paragraphs** (`cr_sim/engine/actions.py:13`, `:15`, `:23`, `:33`, `:41`);
the code wins, and the override is filed at
[`../../_meta/overrides.md`](../../_meta/overrides.md), row 1. What follows is a
pointer to that docstring, not a replacement for it — read `:1-54` for the
worked examples.

1. **Actions are not all in the `ACTION` namespace.** `Graveyard_rework_Group`
   lists twelve sub-actions that resolve only under `EXT`. Resolution tries
   `ACTION` then falls back to `EXT`; an `ACTION`-only lookup finds the group
   and silently drops every skeleton it was supposed to spawn.
2. **Positions are expressions, not numbers** — and they are in milli-tiles
   while the engine is in subtiles, so `ActionContext` converts at the scope
   boundary and nowhere else.
3. **The expression language is small and closed**, and is evaluated by
   rewriting `&&`/`||`/`!`, parsing with `ast`, and walking against a node
   whitelist — **never `eval`**, which would run arbitrary code out of a data
   file. An unknown function name is an error rather than a silent zero.
4. **An action may be written out in place instead of referenced.** Nineteen
   nodes in the build are inline dicts — Dark Magic's whole effect among them —
   and every field that can hold an action can hold either spelling. Accepting
   only names is exactly how Dark Magic shipped as a 5-elixir spell that did
   nothing.

**Coverage is deliberately partial and deliberately loud.** Every class type
that is not implemented is counted in `unsupported`, along with bracketed
non-class gaps — `<condition:...>`, `<select:rand(n)>`,
`<changedata:projectile>`, `<missing:...>`. The engine's gaps are enumerable
rather than discovered one card at a time. An unreadable `Condition` is
recorded and treated as **open**, because a gate nobody can read should not
silently disable a card.

**Four of those gaps are deliberate, and each has its own card** — a
`deliberate ghost` is a tripwire, and implementing one without reading its
reason removes it: [`action-taunt.md`](action-taunt.md),
[`action-change-data-projectile.md`](action-change-data-projectile.md),
[`action-select-rand.md`](action-select-rand.md),
[`action-ground-to-air.md`](action-ground-to-air.md). The reasons are written
together at `cr_sim/engine/actions.py:56-82`.

Battle callbacks are **injected, never imported**, so this module does not
import `battle` and the two can be tested apart.

## Shape

- `ActionInterpreter(data, clock, spawn, clone, place_area, count_units,
  apply_buff, *, nearby, set_grounded, change_data, deal_damage, arm_counter,
  fire_projectile)` — 13 callbacks, each defaulting to a no-op.
- `resolve` (cached) → `start` (honours the entry node's own `ActionDelay`) →
  `run` → `_passes` → a handler → `schedule` → `drain`.
- `_HANDLERS` — **22 `ClassType` keys mapping to 18 distinct handlers**
  (`_handle_next` serves four keys, `_handle_spawn` two). Anything absent lands
  in `unsupported`.
- `COSMETIC_CLASSES` (`:456`) — returned from silently rather than counted.
- `ActionContext` — team, position in subtiles, `source`, `variables`;
  `expression_scope` exposes 11 names including `has_data` and `get_radius`,
  which ask about the entity the action is *running on* (the victim for a
  per-target action, not the caster).
- Four **deliberate ghosts**, each with its reason at the declining site
  (`:56-82`): `ActionTaunt` (no taunt/lock exists to clear — left absent so a
  future taunt trips the coverage gate rather than inheriting a silent no-op),
  `ActionChangeGameObjectData` with `NewProjectileData`, `ActionSelect` with
  `Condition = "rand(n)"` ([action-select.md](action-select.md)), and
  `ActionGroundToAir` (only user is `WizardHero`, whose graph is unwired).

Citations: `cr_sim/engine/actions.py:482`, `:543`, `:565`, `:576`, `:597`,
`:623`, `:641`, `:1548-1570`, `:537`, `:306`, `:320-340`, `:197`, `:115`,
`:145-157`, `:1-82`; `cr_sim/engine/battle.py:333-347` (the injection),
`:925-933` (`_phase_run_actions`), `:2651` (`_begin_actions`).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `ActionContext`, `evaluate_expression`, `ExpressionError`,
  `matches_filter`, `_Pending`, every `_handle_*`,
  [action-select.md](action-select.md).
- **owned-by:** [../build/logic-data.md](../build/logic-data.md) — the
  `ACTION`/`EXT` fallback is a `LogicData` behaviour this module depends on.
- **joins:** [battle.md](battle.md) (13 injected callbacks; `run_actions` is
  phase 7 of 20), [battle-clone.md](battle-clone.md) (`_cache` and
  `unsupported` are memoised as shared — neither is state the simulation reads
  back), [buff-percent.md](buff-percent.md) (most modern buffs arrive here).
- **looks-like-but-is-not:** `unsupported` is a `Counter`, not an error log —
  a non-zero entry means "this class type was reached", which for a
  deliberate ghost is the expected state, not a regression.

## If you change this

- **Hits:** every card whose behaviour is a graph rather than a column — which
  is most of what has been added to the game in years; `unsupported`'s keys,
  which are the repo's inventory of what is not implemented; and the delay
  queue's ordering, since `drain` partitions on `due_tick <= tick` and a
  dead instigator's queue is abandoned (a destroyed hut must stop producing).
- **Does not hit:** the stat columns. Adding a handler cannot change a
  `UnitSpec`; conversely a card that is wrong on hitpoints is not an action
  problem. And it does not hit the engine's randomness: this interpreter is
  given **no** random stream on purpose ([rng.md](rng.md)) — the obvious wrong
  fix for a `<select:rand(n)>` count is to hand it `battle.rng`.

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_actions.py`, `tests/test_evolutions.py`, `tests/test_death_and_spawners.py` | pin |
| `cr_sim/data/interactions.py` → `KNOWN_UNMAPPED` | names the cards this module owns rather than a column |
| `ActionInterpreter.unsupported` | the coverage gate; read by nothing automated |

## See

- Source: `cr_sim/engine/actions.py`
