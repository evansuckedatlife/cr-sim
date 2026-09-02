---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/battle.py
---

# BattleConfig

The frozen description of a match before it starts: seed, tick rate, both
decks, both evolution slates, two level numbers, and frame recording.

## Why this shape

Frozen because a battle's identity is `(seed, configuration, command list)` —
`cr_sim/replay.py:3` states it as the whole determinism claim. A mutable config
would make a replay a statement about what the config *was*, which is not
something a file can carry.

**The two level fields are the point of this card.** `level` is a *displayed*
1-15 level, rarity-independent, converted per rarity by
`RarityScale.internal_level` at `cr_sim/engine/battle.py:526`. `tower_level` is a
**raw** tower level fed straight to `TowerScale` at `:470`, with no display
conversion anywhere. Different ladder, different arithmetic, both default to
11, and neither is validated against the other.

That default is bug 1. `--tower-level` reached the local probe's `_env()` and
was omitted from the `VecEnvConfig` used under `--workers`, which defaults it
to 11 — so every worker trained at level 11 while `config.json` recorded 5 and
the probe evaluated at 5. About 90% of battles drew and the agent learned from
shaping alone. The plumbing was fixed with the reason written in place at
`cr_sim/train/run.py:996-1009`, but **the shape is still live**: `skip_forced`
has no `VecEnvConfig` field at all, so it can never be anything but its default
under `--workers`. Adding a field here is not done until it exists on every
construction path — see [../measurement/config-json.md](../measurement/config-json.md)
for what a run actually records.

`record_frames` is the one field a projection must override rather than
inherit, and [lookahead.md](lookahead.md) says why.

## Shape

- `seed`, `ticks_per_second=60`, `blue_deck` / `red_deck`,
  `blue_evolutions` / `red_evolutions` (empty by default — having an evolution
  *available* is not the same as slotting it), `level=11`, `tower_level=11`,
  `record_frames=False`, `frame_interval=3`.
- Read at exactly six engine sites: the clock
  (`cr_sim/engine/battle.py:326`), the seed (`:332`), the decks (`:433-454`),
  `tower_level` (`:470`), `level` (`:526`) and frames (`:771`).
- `BattleResult` — `winner`, `blue_crowns`, `red_crowns`, `ticks`, `reason`.

Citations: `cr_sim/engine/battle.py:210`, `:221`, `:222`, `:231`, `:470`,
`:526`; `cr_sim/replay.py:3`; `cr_sim/train/run.py:996-1009`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `BattleResult`.
- **owned-by:** whoever constructs the battle — `cr_sim/api/env.py`,
  `cr_sim/api/vec.py`, `cr_sim/cli.py:357`, `cr_sim/play/session.py`,
  `cr_sim/data/interactions.py`.
- **joins:** [../build/card-ladder.md](../build/card-ladder.md) (`level`),
  [../build/tower-ladder.md](../build/tower-ladder.md) (`tower_level`),
  [../build/tick-clock.md](../build/tick-clock.md),
  [rng.md](rng.md) (`seed`), [replay.md](replay.md).
- **looks-like-but-is-not:** `Replay`. `Replay.to_json` persists **neither
  `tower_level` nor the evolutions** (`cr_sim/replay.py:127-140`), despite the
  module claiming a battle is fully described by seed, configuration and
  commands. A saved replay is not a saved config.

## If you change this

- **Hits:** every construction path, and they are not one list — the local
  `_env()`, the `VecEnvConfig` used by workers, the CLI, the play server, and
  the interaction harness. A field added here reaches the engine the moment it
  is read; it reaches a *training run* only when every one of those paths
  passes it. `Replay.to_json` too, which will silently keep omitting it.
- **Does not hit:** the observation. Nothing in the encoding reads a
  `BattleConfig`, so a level change produces no shape change, no version bump
  and no `check_observation` failure — the run just trains against a different
  game. The obvious wrong reassurance is "the observation contract still
  matches"; it always will.

## Surfaces

| Surface | Role |
|---|---|
| `cr-sim battle` (`cr_sim/cli.py:541-554`) | writes — and has **no** `--tower-level` flag, so every CLI battle runs towers at 11 |
| `cr_sim/api/env.py`, `cr_sim/api/vec.py` (`VecEnvConfig`) | write — the two paths bug 1 lived between |
| `config.json` in a run directory | records — and recorded 5 while workers ran 11 |
| `cr_sim/play/session.py` | writes |

## See

- Source: `cr_sim/engine/battle.py`
