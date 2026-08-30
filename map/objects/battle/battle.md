---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/battle.py
---

# Battle

One match: 2 900 lines and the engine's single mutable god-object. Owns the
tick loop, both `Player`s, the entity list, the tower index and every phase.

## Why this shape

**The order of work inside a tick is the mechanics.** Whether targeting runs
before or after movement decides whether a unit that just came into range
attacks this tick or next; whether deaths resolve before or after projectiles
decides whether a dying unit's shot still lands. So the order lives in one
place as a named 20-tuple, `PHASES`, bound to methods once at construction —
not implicit in the sequence of statements in a 200-line `step()`. That makes
"which orderings are load-bearing" a directed search against the interaction
suite rather than guesswork.

`step()` does three things before the phases: rebuild the spatial index, so
every phase in the tick sees one consistent snapshot of where things are;
snapshot both crown counts, so `check_victory` can tell "a tower died *this*
tick" from "a tower is already dead" — the distinction sudden death is built
on; then run the twenty.

`play_card` is the legality check, and it is **deliberately independent of the
action mask** in `cr_sim/api/encoding.py`. Two implementations of one rule,
which is a real risk and a deliberate one: an agent trained against legality
the human path does not honour would be training against a different game.
`fallen_enemy_towers` is public for exactly that reason — both paths must get
the same answer.

**Code wins over the module docstring.** `cr_sim/engine/battle.py:16-18` still calls
targeting, combat, projectiles and collision "stubs that name what they will
do". All four are complete phases — `_phase_acquire_targets` (`:1078`),
`_phase_resolve_attacks` (`:1133`), `_phase_advance_projectiles` (`:1539`),
`_phase_resolve_collisions` (`:2119`). The paragraph is M1-era and wrong ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 4).

## Shape

- `PHASES` — 20 names, `regenerate_elixir` → `check_victory`; bound at `:427`.
- `Battle.__slots__` — 47 slots (`:266-314`); the caches (`_specs`, `_routes`, `_attacks`,
  `_charge`, `_spawn_timers`, ...) are all keyed by entity id.
- `Player` — elixir, `cycle` (hand is the first four), crowns, `last_played`
  for Mirror, `evolutions` + `evolution_charge`. A slot starts *charged*, so
  the first play of an evolution card is the evolved one.
- `play_card` → variant resolution → Mirror → affordability →
  `Arena.can_deploy` → spend → cycle → evolution swap → `_cast` or `_deploy`.
- `_towers` — a per-team index built once in `_spawn_towers` and **never
  pruned**: a destroyed tower stays in it, dead. That is on purpose (`_king`
  and `fallen_enemy_towers` need to see a dead King), and it is why
  `RewardTracker._snapshot` reading `_towers` directly
  (`cr_sim/api/reward.py:217-222`) agrees with `total_tower_hitpoints`
  (`cr_sim/api/encoding.py:506`, which walks `entities + graveyard`) — two
  notions of "the towers" that agree today because both include the dead.
- `_refresh_occupancy` — rebuilds the path grid's building costs only when the
  standing-structure **signature** changes; rebuilding every tick was 40% of
  throughput. The signature keys on entity id.

Citations: `cr_sim/engine/battle.py:239`, `:243-263`, `:427`, `:754`, `:538`,
`:144`, `:356`, `:456-486`, `:1957-1991`, `:2795`, `:2807-2827`, `:16-18`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `Player`, `BattleResult`, the phases,
  [battle-clone.md](battle-clone.md), [battle-config.md](battle-config.md).
- **owned-by:** [../build/logic-data.md](../build/logic-data.md),
  [../build/card.md](../build/card.md),
  [../build/card-ladder.md](../build/card-ladder.md) — a battle is constructed
  from all three plus a config.
- **joins:** every other card in this cluster; upward, `cr_sim/api/env.py` and
  `cr_sim/api/encoding.py` read a `Battle` and nothing writes one.
- **looks-like-but-is-not:** the action mask. `play_card` and
  `legal_action_mask` are two implementations of one rule and can drift.
  `Battle.frames` is not state — never hashed, cosmetic by construction, so
  recording cannot change an outcome (`:410`, `:789`).

## If you change this

- **Hits:** reordering `PHASES` changes gameplay, deliberately — the
  interaction gate ([../build/validation-gates.md](../build/validation-gates.md))
  is the instrument for it, not the unit tests. Adding a `__slots__` entry hits
  [battle-clone.md](battle-clone.md): a new slot defaults to being **copied**,
  which is the safe direction, but a new *immutable cache* left out of
  `_SHARED` silently costs a clone. Changing `play_card`'s legality hits
  `cr_sim/api/encoding.py:586-640`, which must agree.
- **Does not hit:** the specs. Nothing in the loop converts a unit; if a number
  is wrong at tick 4000 it was wrong at tick 0, in
  [../build/unit-spec.md](../build/unit-spec.md). The other wrong next step is
  the reward: `cr_sim/api/reward.py` reads `battle._towers` and
  `battle.damage_log` but writes nothing back, so a phase change cannot be
  debugged from the reward's terms.

## Surfaces

| Surface | Role |
|---|---|
| `cr-sim battle` (`cr_sim/cli.py:357`) | reads / writes an HTML replay |
| `cr_sim/api/env.py`, `cr_sim/api/vec.py` | read — one `Battle` per environment |
| `cr_sim/play/session.py`, `cr_sim/render/web.py` | read — browser play and the viewer |
| `cr_sim/data/interactions.py:798` | **writes** `battle._towers` directly, to strip towers from a duel |
| `tests/test_arena_and_battle.py`, `test_match_rules.py`, `test_mirror.py` | read |

## See

- Source: `cr_sim/engine/battle.py`
