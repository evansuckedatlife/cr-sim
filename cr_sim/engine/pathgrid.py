"""Weighted pathfinding over the arena grid.

Until now a ground unit walked at its target in a straight line and only ever
planned a route to cross the river. That makes a building decorative: a Cannon
dropped in the lane to pull a Giant off its line does not pull anything,
because nothing in the movement code ever considered going around it.

The game ships the costs for this, and they say what the mechanic is::

    PATHFINDING_DEFAULT_COST     8
    PATHFINDING_ROAD_COST        5
    PATHFINDING_BUILDING_COST   50
    PATHFINDING_BLOCKED_COST    50
    PATHFINDING_WATER_COST       7

A building is not a wall. At 50 against a default of 8 it is somewhere it costs
six times as much to travel, so a unit routes around one whenever there is a
way around and pushes through when there is not -- which is exactly how a
building placed in the lane behaves, and is not something a hard obstacle would
reproduce. ``PATHFINDING_DYNAMIC_OCCLUSIONS`` being true is the same statement:
the cost layer changes during a match.

Terrain that cannot be entered at all is left out of the search rather than
given a large cost. A ground unit does not cross the river slowly, it does not
cross it, and a cost model that let it would produce paths no unit can walk.
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, Mapping

from .arena import Arena, Tile

__all__ = [
    "PathGrid", "find_path", "simplify", "flow_field", "next_cell",
    "PATH_COSTS", "load_path_costs",
]

#: Defaults, matching this build's globals. Overridden per build by
#: :func:`load_path_costs` so a rebalance in the files is picked up rather than
#: silently contradicted by a constant here.
PATH_COSTS: dict[str, int] = {
    "default": 8,
    "road": 5,
    "water": 7,
    "blocked": 50,
    "building": 50,
    "heuristic": 5,
}

_GLOBAL_KEYS = {
    "default": "PATHFINDING_DEFAULT_COST",
    "road": "PATHFINDING_ROAD_COST",
    "water": "PATHFINDING_WATER_COST",
    "blocked": "PATHFINDING_BLOCKED_COST",
    "building": "PATHFINDING_BUILDING_COST",
    "heuristic": "PATHFINDING_DEFAULTHEURISTIC_COST",
}

_NEIGHBOURS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

#: Diagonal steps cost this many halves of a straight one. Charging a diagonal
#: the same as a straight step makes eight-way movement cheaper per tile of
#: progress than four-way, and A* answers by producing staircases.
_DIAGONAL_HALVES = 3


def load_path_costs(globals_map: Mapping[str, Any]) -> dict[str, int]:
    """Read the costs from a build's globals, falling back to the defaults."""
    costs = dict(PATH_COSTS)
    for name, key in _GLOBAL_KEYS.items():
        value = globals_map.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            costs[name] = value
    return costs


class PathGrid:
    """Per-cell movement cost, terrain plus whatever is standing on it.

    Terrain is fixed and computed once per arena. Buildings are not: they
    appear and die mid-match, and a path found before one existed must stop
    being reused after. Rather than hunt down and invalidate live routes, the
    grid carries a ``version`` that moves when occupancy does, and cached paths
    are keyed by it -- a stale path is never *found*, it just stops matching.
    """

    __slots__ = (
        "arena", "costs", "_ground", "_air", "_occupied", "version", "_fields",
        "_combined",
    )

    def __init__(self, arena: Arena, costs: Mapping[str, int] | None = None) -> None:
        self.arena = arena
        self.costs = dict(costs or PATH_COSTS)
        self._occupied: dict[int, int] = {}
        self.version = 0
        self._ground = self._terrain(flying=False)
        self._air = self._terrain(flying=True)
        #: Cached distance fields, keyed by (goal, flying, version).
        self._fields: dict[tuple[tuple[int, int], bool, int], list[int]] = {}
        #: Terrain and occupancy folded together, per (flying, version).
        self._combined: dict[tuple[bool, int], list[int]] = {}

    def _terrain(self, *, flying: bool) -> list[int]:
        """Cost to enter each cell, or 0 for cells that cannot be entered."""
        arena = self.arena
        costs = self.costs
        cells: list[int] = []
        for cy in range(arena.half_height):
            for cx in range(arena.half_width):
                bits = arena.cell(cx, cy)
                if bits & Tile.BRIDGE:
                    # The cheapest ground on the board, which is why every push
                    # funnels onto one whether or not it needs to cross.
                    cells.append(costs["road"])
                elif bits & Tile.WATER:
                    cells.append(costs["water"] if flying else 0)
                elif bits & Tile.BLOCKED:
                    cells.append(costs["blocked"] if flying else 0)
                else:
                    cells.append(costs["default"])
        return cells

    # -- dynamic layer -----------------------------------------------------

    def set_occupancy(self, cells: Mapping[int, int]) -> None:
        """Replace the building layer, as ``{cell index: extra cost}``.

        The version only moves when something changed. Buildings are placed a
        handful of times a match, and since the version invalidates every
        cached path, one that moved every tick would switch the cache off.
        """
        if cells == self._occupied:
            return
        self._occupied = dict(cells)
        self.version += 1
        # Every field and every folded cost array describes a board that no
        # longer exists.
        self._fields.clear()
        self._combined.clear()

    def combined(self, flying: bool = False) -> list[int]:
        """Terrain plus occupancy as one flat array, cached per version.

        The search asked ``cost()`` once per neighbour per cell, which came to
        thirty million calls in a short training run -- a Python call and two
        dict lookups where an array index would do. Folding the two layers
        together once per version turns the inner loop into indexing.
        """
        key = (flying, self.version)
        found = self._combined.get(key)
        if found is None:
            base = list(self._air if flying else self._ground)
            for index, extra in self._occupied.items():
                if 0 <= index < len(base) and base[index]:
                    base[index] += extra
            self._combined[key] = base
            found = base
        return found

    def index_of(self, cx: int, cy: int) -> int:
        return cy * self.arena.half_width + cx

    def cost(self, cx: int, cy: int, *, flying: bool = False) -> int:
        """Cost of entering a cell. Zero means it cannot be entered."""
        arena = self.arena
        if not (0 <= cx < arena.half_width and 0 <= cy < arena.half_height):
            return 0
        index = cy * arena.half_width + cx
        base = (self._air if flying else self._ground)[index]
        if base == 0:
            return 0
        return base + self._occupied.get(index, 0)


def find_path(
    grid: PathGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    flying: bool = False,
    limit: int = 3000,
) -> list[tuple[int, int]]:
    """Cheapest path from ``start`` to ``goal`` in grid cells.

    Returns an empty list when there is no route, which is an outcome rather
    than an error: a unit walled in by buildings has nowhere to go and should
    fall back to walking at its target instead of raising.

    ``limit`` bounds the expansion. The grid is only 2304 cells so a full
    search is cheap, but this runs per unit and a pathological case has to
    degrade to a straight line rather than stall a tick.
    """
    if start == goal:
        return [goal]
    if grid.cost(*goal, flying=flying) == 0 or grid.cost(*start, flying=flying) == 0:
        return []

    width, height = grid.arena.half_width, grid.arena.half_height
    scale = grid.costs["heuristic"]

    def estimate(cx: int, cy: int) -> int:
        # Chebyshev, scaled by the cheapest possible step. Admissible because
        # no step costs less than `scale`, which is what keeps A* optimal --
        # an overestimate here would return quick, wrong paths.
        return scale * max(abs(cx - goal[0]), abs(cy - goal[1]))

    open_heap: list[tuple[int, int, tuple[int, int]]] = [(estimate(*start), 0, start)]
    best: dict[tuple[int, int], int] = {start: 0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0

    while open_heap and expanded < limit:
        _priority, spent, current = heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path
        if spent > best.get(current, 1 << 30):
            continue
        expanded += 1
        cx, cy = current
        for dx, dy in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            step = grid.cost(nx, ny, flying=flying)
            if step == 0:
                continue
            if dx and dy:
                # A diagonal may not cut the corner between two cells that are
                # themselves impassable, or a unit slips through a gap the
                # geometry does not have.
                if grid.cost(cx + dx, cy, flying=flying) == 0:
                    continue
                if grid.cost(cx, cy + dy, flying=flying) == 0:
                    continue
                step = step * _DIAGONAL_HALVES // 2
            candidate = spent + step
            if candidate < best.get((nx, ny), 1 << 30):
                best[(nx, ny)] = candidate
                came[(nx, ny)] = current
                heappush(open_heap, (candidate + estimate(nx, ny), candidate, (nx, ny)))
    return []


def simplify(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep only the cells where the path changes direction.

    A* returns one cell per step, and a route walking forty one-cell segments
    pays a distance calculation for each. Only the corners carry information.
    """
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    for previous, current, following in zip(path, path[1:], path[2:]):
        before = (current[0] - previous[0], current[1] - previous[1])
        after = (following[0] - current[0], following[1] - current[1])
        if before != after:
            out.append(current)
    out.append(path[-1])
    return out


UNREACHABLE = 1 << 30

#: Cells a flow field's goal is rounded to. The field was designed around a
#: handful of fixed goals -- everything walks at one of three towers -- but a
#: route is asked for a *target entity's* position, and a moving target crosses
#: a cell every few ticks. Every crossing was a fresh Dijkstra: profiling a
#: training run found 4,716 field builds and 34 million cost lookups, by far
#: the largest single cost in the engine.
#:
#: Rounding the goal to a 2x2 block means everything chasing roughly the same
#: place shares one field. The path is unaffected in any way that survives:
#: the route's last waypoint is replaced with the true goal, so the rounding
#: only decides which way a unit approaches from, at half-tile precision.
GOAL_SNAP = 2

#: Distinct fields kept. Bounded because a match with many separate skirmishes
#: has many goals, and an unbounded cache would hold a field for every one for
#: as long as the occupancy version stood.
MAX_FIELDS = 64


def flow_field(grid: PathGrid, goal: tuple[int, int], *, flying: bool = False) -> list[int]:
    """Cost from every cell to ``goal``, by Dijkstra outward from the goal.

    A* answers one unit's question at a time, and a long path measured at
    11.5ms is far more than a whole engine tick costs. But the questions are
    not independent: every unit on a side is walking at one of three towers, so
    a handful of goals serve a whole board of units.

    So the search runs once per goal and every unit reads its answer. A unit's
    next step is simply the neighbouring cell with the lowest remaining cost,
    which is a lookup rather than a search, and the field is cached until
    occupancy changes.

    Run from the goal outward, so the result is cost-to-goal for every cell
    rather than cost-from-one-start.
    """
    goal = (goal[0] // GOAL_SNAP * GOAL_SNAP, goal[1] // GOAL_SNAP * GOAL_SNAP)
    key = (goal, flying, grid.version)
    cached = grid._fields.get(key)
    if cached is not None:
        return cached
    if len(grid._fields) >= MAX_FIELDS:
        grid._fields.clear()

    width, height = grid.arena.half_width, grid.arena.half_height
    board = grid.combined(flying)
    field = [UNREACHABLE] * (width * height)
    if board[goal[1] * width + goal[0]] == 0:
        grid._fields[key] = field
        return field

    field[goal[1] * width + goal[0]] = 0
    heap: list[tuple[int, int, int]] = [(0, goal[0], goal[1])]
    while heap:
        spent, cx, cy = heappop(heap)
        if spent > field[cy * width + cx]:
            continue
        row = cy * width
        for dx, dy in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            index = ny * width + nx
            step = board[index]
            if step == 0:
                continue
            if dx and dy:
                # A diagonal may not cut the corner between two impassable
                # cells, or a unit slips through a gap the geometry lacks.
                if board[row + cx + dx] == 0 or board[(cy + dy) * width + cx] == 0:
                    continue
                step = step * _DIAGONAL_HALVES // 2
            candidate = spent + step
            if candidate < field[index]:
                field[index] = candidate
                heappush(heap, (candidate, nx, ny))

    grid._fields[key] = field
    return field


def next_cell(
    grid: PathGrid, field: list[int], cell: tuple[int, int]
) -> tuple[int, int] | None:
    """The neighbouring cell that gets closest to the field's goal.

    None when the unit is standing on the goal or nothing adjacent improves --
    the latter meaning it is walled in, which the caller handles by walking
    straight at its target rather than by standing still.
    """
    width, height = grid.arena.half_width, grid.arena.half_height
    cx, cy = cell
    here = field[cy * width + cx] if 0 <= cx < width and 0 <= cy < height else UNREACHABLE
    if here == 0:
        return None
    best_cost, best = here, None
    for dx, dy in _NEIGHBOURS:
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        value = field[ny * width + nx]
        if value < best_cost:
            best_cost, best = value, (nx, ny)
    return best


def field_path(
    grid: PathGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    flying: bool = False,
    limit: int = 512,
) -> list[tuple[int, int]]:
    """Walk a flow field from ``start`` to ``goal``, as cells.

    ``limit`` bounds the walk. A correct field is strictly decreasing so it
    cannot loop, but a bound means a corrupted one degrades to a short path
    rather than to a hung tick.
    """
    field = flow_field(grid, goal, flying=flying)
    goal = (goal[0] // GOAL_SNAP * GOAL_SNAP, goal[1] // GOAL_SNAP * GOAL_SNAP)
    cell, path = start, [start]
    for _ in range(limit):
        step = next_cell(grid, field, cell)
        if step is None:
            break
        path.append(step)
        cell = step
        if cell == goal:
            break
    return path
