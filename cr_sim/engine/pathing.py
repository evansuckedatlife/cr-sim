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
from .fixed import distance, point_along

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
) -> Route:
    """Build a route from ``start`` to ``goal``.

    Flying units go straight -- ignoring the river is the entire value of
    flight. Ground units crossing the river are sent through the nearer bridge,
    entering and leaving it at its centre so they funnel the way real troops do.
    """
    route = Route()
    if flying or not _crosses_river(arena, start[1], goal[1]):
        route.waypoints = [goal]
    else:
        top, bottom = arena.river_band()
        _left, _right, centre_x = arena.nearest_bridge(start[0])
        going_up = goal[1] > start[1]
        near, far = (top, bottom) if going_up else (bottom, top)
        route.waypoints = [(centre_x, near), (centre_x, far), goal]
    route.start(start)
    return route


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
