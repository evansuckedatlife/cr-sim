---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/pathgrid.py
---

# Pathing

How a ground unit gets from where it is to what it is walking at: a weighted
cost grid, one Dijkstra field per goal shared by every unit heading there, and
a waypoint route on top. Plus the spatial index every neighbourhood query runs
through.

## Why this shape

**A building is not a wall.** The game ships the costs and they say what the
mechanic is: default 8, road 5, water 7, building 50, blocked 50. At 50 against
8 a building is somewhere it costs six times as much to travel, so a unit
routes *around* one when there is a way around and pushes through when there is
not — which is exactly how a building placed in the lane behaves and is not
something a hard obstacle reproduces. Terrain that cannot be entered at all is
left out of the search rather than given a large cost: a ground unit does not
cross the river slowly, it does not cross it.

**One field per goal, not one search per unit.** A* answers one unit's question
at a time and a long path measured 11.5ms — more than a whole tick costs. But
the questions are not independent: everything on a side walks at one of three
towers. So Dijkstra runs *outward from the goal* and every unit reads its
answer as a lookup. `GOAL_SNAP = 2` rounds the goal to a 2x2 block so everything
chasing roughly the same place shares one field — profiling found 4 716 field
builds and 34 million cost lookups before that, by far the largest single cost
in the engine. The rounding is safe because the route's last waypoint is
replaced with the true goal.

**Staleness is handled by versioning, not invalidation.** Buildings appear and
die mid-match, so `PathGrid.version` moves when occupancy does and fields are
keyed by it. A stale field is never *found*; it simply stops matching.

**The spatial index is rebuilt from scratch every tick.** Incremental updates
are faster in principle and a known source of heisenbugs — one missed move and
an entity is queryable at a position it left minutes ago. Rebuilds cannot
drift. Only the buckets left dirty by the last rebuild are cleared; clearing
all of them was about a third of a tick.

## Shape

- `PATH_COSTS` / `load_path_costs(globals_map)` — defaults overridden per build,
  so a rebalance in the files is picked up rather than contradicted.
- `PathGrid` — `_ground` / `_air` terrain layers computed once, `_occupied`
  overlay, `version`, `_fields`, `_combined`; its own `__deepcopy__`.
- `flow_field(grid, goal, flying=False)` — `MAX_FIELDS = 64`, cleared wholesale
  when full. Diagonals cost `_DIAGONAL_HALVES = 3` halves of a straight step
  and may not cut a corner between two impassable cells.
- `Route` / `route_to` — what the engine calls. Flying goes straight; a clear
  line needs no plan; otherwise `field_path` → `simplify` → waypoints, last one
  replaced with the true goal. Without a grid it falls back to "straight,
  through the nearer bridge".
- `SpatialIndex` — uniform grid, `DEFAULT_CELL` = 2 tiles (measured against
  1-tile cells, which lose), rebuilt in `Battle.step` before any phase.
- `find_path` (A*) — **leftover**. Superseded by `flow_field`, kept alive by
  `tests/test_pathfinding.py`; `costs["heuristic"]` is read nowhere else.

Citations: `cr_sim/engine/pathgrid.py:43`, `:72`, `:82`, `:69`, `:216`,
`:321`, `:326`, `:329`, `:390`, `:415`, `:290`, `:1-26`;
`cr_sim/engine/pathing.py:39`, `:105`, `:162`, `:222`, `:239`;
`cr_sim/engine/spatial.py:33`, `:36`, `:46-59`; `cr_sim/engine/battle.py:760`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `PathGrid`, `flow_field`, `Route`, `SpatialIndex`, `PATH_COSTS`.
- **owned-by:** [arena.md](arena.md) (terrain) and
  [../build/logic-data.md](../build/logic-data.md) (`globals.csv` costs).
- **joins:** [battle.md](battle.md) — `_refresh_occupancy` is the only writer of
  occupancy and it keys on entity ids; `_phase_move_units` is the only caller of
  `route_to`; `step` rebuilds the index. [targeting.md](targeting.md) reads the
  index through `in_reach`.
- **looks-like-but-is-not:** `pathing.py`'s module docstring still describes
  M3 weighted pathfinding as future work and the module as "the skeleton"
  (`cr_sim/engine/pathing.py:8-12`). The grid path is live and is the default; **code wins** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 5).
  `next_cell` / `field_path` / `simplify` are live helpers, not the entry point
  — `route_to` is.

## If you change this

- **Hits:** every ground unit's path, so the interaction gate's simulated
  matrix and every stored hash stream; throughput, which gates training — this
  was the largest engine cost and `_refresh_occupancy`'s signature test was 40%
  of it; and `Battle._routes`, keyed by entity id and holding a `Route` whose
  waypoints were computed under an older `version`.
- **Does not hit:** targeting. A unit's *target* is chosen by
  [targeting.md](targeting.md) from the spatial index; pathing only decides how
  it walks there. A unit walking past an enemy is a targeting question
  (building-targeters cannot see troops), not a routing one — that is the wrong
  file to open for it. It also does not hit flying units, which take a straight
  line and never touch the grid.

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_pathfinding.py` | the only thing keeping `find_path` alive |
| `scripts/bench_engine.py` | reads — throughput measurement |
| `cr_sim/render/web.py` | none — the viewer draws positions, not routes |

## See

- Source: `cr_sim/engine/pathgrid.py`, `cr_sim/engine/pathing.py`, `cr_sim/engine/spatial.py`
