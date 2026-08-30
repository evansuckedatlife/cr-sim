---
type: object
cluster: battle
universe: leftover
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/replay.py
---

# Replay

The serialisable form of a match: seed, tick rate, decks, levels, command list,
and optionally per-tick hashes and viewer frames. **Leftover** — constructed by
no production code in `cr_sim/`.

## Why this shape

The module's opening claim is the engine's whole contract: *a battle is fully
described by `(seed, configuration, command list)`. Nothing else may influence
the outcome — no wall clock, no dict iteration order, no floating point.*
`Command` is the only input a battle takes from outside a tick, which is what
makes a five-field record enough to reproduce a match.

Hashes are optional because storing one per tick costs about 85KB a battle and
is only needed while verifying. Frames are optional because they are cosmetic
and never hashed.

**Code wins over that claim.** `to_json` persists `seed`, `ticks_per_second`,
`build`, `decks`, `levels` and `commands` — and **not `tower_level`, not the
evolution slates**. A `Replay` therefore does not carry enough to rebuild the
`BattleConfig` that produced it, so a "reproduced" battle can silently run its
towers at the default 11 ([battle-config.md](battle-config.md), bug 1's exact
shape). The claim at `:3` is the one to fix, or the serialiser is ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 7).

`Command` itself is **live** and is not leftover: the engine imports it
(`cr_sim/engine/battle.py:31`), `Battle.queue` holds a list of them, and
`_phase_apply_commands` drains them.

## Shape

- `Command(tick, team, card, x, y)` — frozen, slotted, with `as_json` /
  `from_json`. Live.
- `Replay` — the container plus `add`, `commands_for_tick`, `by_tick`,
  `to_json`, `save`, `load`. Leftover: only `tests/test_engine_core.py:399-402`
  builds one.
- `DivergenceError` — exported, raised nowhere.
- `compare_hashes` is owned by [state-hash.md](state-hash.md).

Citations: `cr_sim/replay.py:73`, `:97`, `:127-140`, `:3-6`, `:34`;
`cr_sim/engine/battle.py:31`, `:535-536`, `:857-865`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `Command`.
- **owned-by:** nothing constructs a `Replay` in production; the HTML viewer
  path writes its own JSON from `Battle.frames`
  (`cr_sim/render/web.py`, `cr_sim/cli.py:357`).
- **joins:** [state-hash.md](state-hash.md), [battle.md](battle.md),
  [battle-config.md](battle-config.md).
- **looks-like-but-is-not:** the HTML replay `cr-sim battle --html` writes.
  That is a frame dump for the viewer, not a `Replay`, and it cannot be
  replayed — only watched. Two different artefacts called "replay".

## If you change this

- **Hits:** `tests/test_engine_core.py` only, plus anyone who starts using
  `Replay` on the strength of its docstring. Adding `tower_level` and the
  evolutions to `to_json` / `load` is the change this card exists to point at.
- **Does not hit:** the engine or any training run. Nothing imports `Replay`
  outside tests, so extending it fixes no live bug — the same *omission* in
  `VecEnvConfig` is what shipped bug 1, and that is
  [battle-config.md](battle-config.md)'s problem, not this file's. Fixing
  `Replay` and believing bug 1's shape is closed is the wrong conclusion to
  draw here.

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_engine_core.py:398-404` | the only constructor |
| `cr-sim battle --html` (`cr_sim/cli.py:357`) | writes a *different* artefact |
| a human reading `cr_sim/replay.py:3` | reads a claim the serialiser does not keep |

## See

- Source: `cr_sim/replay.py`
