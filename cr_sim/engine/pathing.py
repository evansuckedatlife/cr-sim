"""Ground routing across the river.

Clash Royale's defining spatial constraint is that ground troops can only cross
at two bridges. Everything about how a push develops -- why units funnel, why
a building placed mid-court pulls a whole lane, why the bridge is where fights
happen -- follows from that one rule.

Full weighted-grid pathfinding comes in M3, where the ``PATHFINDING_*`` costs
the game ships (``DEFAULT=8``, ``ROAD=5``, ``WATER=7``, ``BLOCKED=50``,
``BUILDING=50``) start to matter for how units flow around each other. What is
here is the skeleton that constraint requires: a route is a short list of
waypoints, and crossing the river inserts the bridge.

Routes are expressed as waypoints rather than a per-tick direction because
movement is *derived* from distance travelled along a segment (see
:func:`cr_sim.engine.fixed.point_along`), which is what keeps positions from
accumulating rounding drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .arena import Arena
from .fixed import SUBTILES_PER_TILE, distance, point_along
from .pathgrid import PathGrid, field_path, simplify

_HALF_TILE = SUBTILES_PER_TILE // 2

#: How much of the approach is exempt from the blocked test, in subtiles. Wide
#: enough to clear a tower's own footprint, which is what a unit attacking one
#: is walking into on purpose.
_GOAL_SLACK = SUBTILES_PER_TILE * 2

__all__ = ["Route", "route_to", "crosses_river", "step_towards"]


@dataclass(slots=True)
class Route:
    """An ordered list of waypoints plus how far along it a unit has walked."""

    waypoints: list[tuple[int, int]] = field(default_factory=list)
    index: int = 0
    travelled: int = 0
    #: Cached length of the segment currently being walked.
    segment_length: int = 0
    origin: tuple[int, int] = (0, 0)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def target(self) -> tuple[int, int] | None:
        if self.finished:
            return None
        return self.waypoints[self.index]

    def start(self, position: tuple[int, int]) -> None:
        self.origin = position
        self.travelled = 0
        goal = self.target
        self.segment_length = 0 if goal is None else distance(*position, *goal)

    def advance(self, position: tuple[int, int], step: int) -> tuple[int, int]:
        """Walk ``step`` subtiles along the route and return the new position.

        Leftover distance carries into the next segment rather than being
        discarded at each waypoint: a unit rounding the bridge should not lose a
        fraction of a tick's movement every time it passes a corner.
        """
        if self.finished or step <= 0:
            return position

        remaining = step
        current = position
        while remaining > 0 and not self.finished:
            goal = self.waypoints[self.index]
            if self.segment_length <= 0:
                self.origin = current
                self.segment_length = distance(*current, *goal)
                self.travelled = 0
            if self.segment_length <= 0:
                self.index += 1
                self.segment_length = 0
                continue

            left = self.segment_length - self.travelled
            if remaining < left:
                self.travelled += remaining
                remaining = 0
                current = point_along(
                    *self.origin, *goal, self.travelled, self.segment_length
                )
            else:
                remaining -= left
                current = goal
                self.index += 1
                self.segment_length = 0
                self.travelled = 0
                self.origin = current
        return current


def route_to(
    arena: Arena,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    flying: bool = False,
    grid: "PathGrid | None" = None,
) -> Route:
    """Build a route from ``start`` to ``goal``.

    Flying units go straight -- ignoring the river is the entire value of
    flight.

    Ground units follow a weighted flow field when a :class:`PathGrid` is
    given, which is what lets a building bend a push: at a cost of 50 against a
    default of 8, going around one is cheaper than going through unless there
    is no way around. Without a grid the old behaviour stands -- straight,
    through the nearer bridge when the river is in the way -- because the grid
    is optional and a caller without one should still get a usable route rather
    than an exception.
    """
    route = Route()
    if flying:
        route.waypoints = [goal]
        route.start(start)
        return route

    # A clear line needs no plan. Checked here as well as by the caller,
    # because a route asked for over open ground would otherwise be walked as
    # a chain of cell-centre waypoints -- longer to follow, and no straighter.
    if grid is not None and not line_blocked(grid, start, goal):
        route.waypoints = [goal]
        route.start(start)
        return route

    if grid is not None:
        cells = field_path(grid, _to_cell(start), _to_cell(goal))
        if len(cells) > 1:
            waypoints = [_to_world(c) for c in simplify(cells)[1:]]
            # The field lands on a cell centre; the caller asked for a point.
            waypoints[-1] = goal
            route.waypoints = waypoints
            route.start(start)
            return route

    if not _crosses_river(arena, start[1], goal[1]):
        route.waypoints = [goal]
    else:
        top, bottom = arena.river_band()
        _left, _right, centre_x = arena.nearest_bridge(start[0])
        going_up = goal[1] > start[1]
        near, far = (top, bottom) if going_up else (bottom, top)
        route.waypoints = [(centre_x, near), (centre_x, far), goal]
    route.start(start)
    return route


def line_blocked(
    grid: "PathGrid",
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    flying: bool = False,
    samples: int = 8,
) -> bool:
    """Whether walking straight at ``goal`` runs into terrain or a building.

    Pathfinding is for when the direct route fails, and the direct route
    usually does not: two units in an open lane want a straight line, and
    planning a path for that is work with no output. So the cheap question is
    asked first, and the flow field is consulted only when the answer is yes.

    Expensive counts as blocked, not only impassable. A building is cost 50
    against a default of 8 -- deliberately passable, so a unit with nowhere
    else to go still gets through -- so a test that only looked for cost zero
    would never fire for the one obstacle this exists to handle. Anything at or
    above the building cost is worth planning around.

    Sampled rather than traced. A handful of points along the segment catches
    anything a unit could not walk through -- a building is at least a tile
    across and the samples are closer together than that -- and costs a
    fraction of walking every cell the line touches, on a test that runs for
    every moving unit every tick.
    """
    if flying:
        return False
    expensive = grid.costs["building"]
    span = distance(*start, *goal)
    if span <= 0:
        return False
    # The last stretch is not sampled. A unit walking at a building has that
    # building in the occupancy map, so sampling all the way in makes every
    # target its own obstacle -- which had Knights curving around the tower
    # they were attacking and walking 4.79 tiles in the five seconds their
    # speed says is exactly five.
    reach = min(samples - 1, max(1, int(samples * (1 - _GOAL_SLACK / max(span, 1)))))
    for step in range(1, reach + 1):
        x = start[0] + (goal[0] - start[0]) * step // samples
        y = start[1] + (goal[1] - start[1]) * step // samples
        cost = grid.cost(x // _HALF_TILE, y // _HALF_TILE)
        if cost == 0 or cost >= expensive:
            return True
    return False


def _to_cell(point: tuple[int, int]) -> tuple[int, int]:
    return point[0] // _HALF_TILE, point[1] // _HALF_TILE


def _to_world(cell: tuple[int, int]) -> tuple[int, int]:
    """The centre of a cell, so a route does not hug cell corners."""
    return (
        cell[0] * _HALF_TILE + _HALF_TILE // 2,
        cell[1] * _HALF_TILE + _HALF_TILE // 2,
    )


def step_towards(
    start: tuple[int, int], goal: tuple[int, int], step: int
) -> tuple[int, int]:
    """Move ``step`` subtiles straight at ``goal``.

    Chasing a moving target does not want a route. A route is a plan, and a
    plan whose destination moves every tick has to be rebuilt every tick --
    which is exactly what it was doing, allocating a fresh Route per chasing
    unit per tick for a path that is a straight line anyway. Routes are for the
    one thing that genuinely needs planning: crossing the river.
    """
    gap = distance(*start, *goal)
    if gap <= step or gap == 0:
        return goal
    return point_along(*start, *goal, step, gap)


def crosses_river(arena: Arena, start_y: int, goal_y: int) -> bool:
    """Whether a straight line between these two points would cross the water."""
    return _crosses_river(arena, start_y, goal_y)


def _crosses_river(arena: Arena, start_y: int, goal_y: int) -> bool:
    """Whether getting from ``start_y`` to ``goal_y`` involves the water.

    Note the third case. Asking only whether the two *ends* sit on opposite
    banks silently excuses a unit that is already **on a bridge**: mid-crossing
    its own y is inside the band, so every target reads as "same side", it
    abandons the route, steers straight -- and walks diagonally off the edge of
    the bridge into the river. That is exactly how ground troops ended up
    swimming.
    """
    top, bottom = arena.river_band()
    if top == bottom:
        return False
    start_inside = top <= start_y <= bottom
    goal_inside = top <= goal_y <= bottom
    if start_inside and not goal_inside:
        return True
    return (start_y < top and goal_y > bottom) or (start_y > bottom and goal_y < top)
