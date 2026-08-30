---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/arena.py
---

# Arena

Terrain plus structure placement, frozen, with the river and bridge geometry
cached. Read from the game's own `tilemaps/tilemap.csv`, not guessed.

## Why this shape

The tilemap is a 36x64 grid of **half-tiles** over the 18x32 tile board, each
cell a bitfield. Every value in the shipped file decomposes cleanly into the
flags in `Tile` with no leftover bits — that exhaustiveness is what confirms
the reading rather than a plausible interpretation of it.

**Two coordinate systems meet here and confusing them is the easy mistake.**
Cells are indexed 0..35 by 0..63; positions — both `spawn_groups.toml` and
everything the engine does — are grid *lines*. The King Tower's `x = 18` is the
line at 18 half-tiles, tile 9.0, dead centre. Read as a cell index it is 9.25
and the tower sits slightly off-centre forever. The unit rule lives once in
[../../CONTEXT.md](../../CONTEXT.md).

The geometry falls out and all of it checks: the river spans cell rows 30-33
(tiles y 15→17, centred on 16, exactly half of 32); two bridges at cells x 5-8
and 27-30, each two tiles wide, centred on tiles 3.5 and 14.5 — which are
exactly the Princess Tower x positions. Towers sit in line with their bridge.

`spawn_groups.toml` lists **one side only**; the opponent mirrors through
`y → half_height - y`. That is why the file names three objects for a
two-player match.

The derived geometry is cached because `crosses_river` sits under `river_band`
and runs once per moving troop per tick; `river_rows` used to rescan all 2304
cells on every call.

## Shape

- `Arena` — frozen: `cells`, `half_width`, `half_height`, `towers`, `source`,
  plus two non-comparing cached fields.
- `Tile(IntFlag)` — NONE, LANE_LEFT 1, LANE_RIGHT 2, BLOCKED 16, WATER 32,
  MARKER 128, BRIDGE 256, CENTRE 512.
- `_WATER` / `_BLOCKED` / `_LANE_MASK` — **plain-int mirrors**. `int & IntFlag`
  dispatches to the flag's reflected `__rand__` and reconstructs a `Flag`
  instance on every test; `is_walkable` alone was measured spending
  microseconds per call in that reconstruction. Same bit pattern, same result,
  `int.__and__` instead.
- `TowerPlacement(name, team, x, y)` — subtiles, both teams, mirrored.
- `can_deploy(...)` takes `fallen_enemy_towers` — the one piece of battle state
  the arena is handed, so a destroyed Princess Tower expands the deploy zone
  for both `play_card` and the action mask.
- `load_arena(data, tilemap=None, spawn_group="King_PrincessTowers")`.

Citations: `cr_sim/engine/arena.py:109`, `:57`, `:79-95`, `:99`, `:293-345`,
`:440-468`, `:473`, `:5-25`, `:118-135` (the cached geometry).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `Tile`, `TowerPlacement`, `load_arena`.
- **owned-by:** [../build/logic-data.md](../build/logic-data.md)
  (`SPAWN_GROUP`) and `data_cache/.../tilemaps/tilemap.csv`, which is **not**
  in `csv_logic` — `load_arena` reaches a sibling directory (`:481`).
- **joins:** [pathing.md](pathing.md) (`PathGrid` is built from `cells`),
  [battle.md](battle.md) (`_spawn_towers` walks `arena.towers`),
  [../build/subtile.md](../build/subtile.md) (`half_tiles`).
- **looks-like-but-is-not:** `Tile.MARKER` and `Tile.CENTRE` are **leftover** —
  decoded for exhaustiveness, read by no engine code. `Tile.BRIDGE` *is* read
  (`cr_sim/engine/pathgrid.py:147`). `Arena.is_road`
  (`cr_sim/engine/arena.py:212`) is a **ghost**, never called.
  Arena identity is cosmetic, not geometric: 141 of the build's 158 arena rows
  point at the same tilemap.

## If you change this

- **Hits:** `PathGrid._terrain`, and therefore every cached flow field
  (`cr_sim/engine/pathgrid.py:140-175`); `Battle._spawn_towers`, so both sides' tower
  positions and count; `Arena.can_deploy` and the action mask built on it
  (`cr_sim/api/encoding.py:586`); the observation's terrain channel, which is
  pre-scaled and must stay **last**, outside the normalisation slices.
- **Does not hit:** the tower's stats. Placement and scaling are unrelated —
  a tower's hitpoints come from [../build/tower-ladder.md](../build/tower-ladder.md)
  via `build_tower_spec`, and moving a tower does not change them. Nor does it
  hit `spawn_groups.toml`: the mirror is computed here, so adding the second
  side to the data file would double every structure.

## Surfaces

| Surface | Role |
|---|---|
| `cr-sim arena` (`cr_sim/cli.py:468`) | reads |
| `cr_sim/mumu/geometry.py` | reads — maps screen pixels onto this grid |
| `cr_sim/render/web.py`, `cr_sim/play/page.py` | read — draw the board |
| `reference/anchors.json` → `river_top_tile`, `bridge_centre_tiles`, `king_tower_tile` | pins |

## See

- Source: `cr_sim/engine/arena.py`
