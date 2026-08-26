"""Integer fixed-point geometry.

Floating point would make the simulator non-reproducible: results would depend
on the platform's rounding, replays would drift, and two runs of the same seed
could disagree. Everything positional is therefore an ``int``.

The unit is **1/18000 of a tile**, called a *subtile*. That number is not
arbitrary -- it is the smallest one that makes both conversions the engine cares
about exact:

*   Game files store distances in milli-tiles (1/1000 tile), so a milli-tile is
    exactly ``18`` subtiles.
*   Game files store speeds in tiles per minute. At 60 ticks per second there
    are 3600 ticks in a minute, so a unit moving ``Speed`` tiles/min covers
    ``Speed * 18000 / 3600 = Speed * 5`` subtiles per tick -- exactly, with no
    remainder. At 20 ticks/second it is ``Speed * 15``, also exact, which is why
    the tick rate can be lowered for cheap training runs without changing
    behaviour.

Distances are compared squared wherever possible; an 18x32 tile arena is
576,000 subtiles on its long axis, so a squared distance fits comfortably in
the 64-bit range Python ints handle natively.
"""

from __future__ import annotations

import math

__all__ = [
    "SUBTILES_PER_TILE",
    "SUBTILES_PER_MILLI_TILE",
    "tiles",
    "milli_tiles",
    "half_tiles",
    "to_tiles",
    "distance_squared",
    "distance",
    "circles_overlap",
    "within_range",
    "point_along",
    "clamp",
]

#: Subtiles in one arena tile. See the module docstring for why 18000.
SUBTILES_PER_TILE = 18_000

#: Game files use milli-tiles; this is the exact conversion factor.
SUBTILES_PER_MILLI_TILE = 18


def tiles(value: float) -> int:
    """Tiles -> subtiles. Accepts a float for readability in constants only."""
    return round(value * SUBTILES_PER_TILE)


def milli_tiles(value: int) -> int:
    """Milli-tiles (the unit in every game file) -> subtiles. Exact."""
    return value * SUBTILES_PER_MILLI_TILE


def half_tiles(value: int) -> int:
    """Half-tiles (the unit spawn_groups.toml uses) -> subtiles. Exact."""
    return value * (SUBTILES_PER_TILE // 2)


def to_tiles(value: int) -> float:
    """Subtiles -> tiles. For display and tests only, never for engine logic."""
    return value / SUBTILES_PER_TILE


def distance_squared(ax: int, ay: int, bx: int, by: int) -> int:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def distance(ax: int, ay: int, bx: int, by: int) -> int:
    """Euclidean distance, truncated to a whole subtile.

    ``math.isqrt`` is exact integer square root -- no float involved, so this is
    reproducible everywhere.
    """
    return math.isqrt(distance_squared(ax, ay, bx, by))


def within_range(ax: int, ay: int, bx: int, by: int, reach: int) -> bool:
    """Is B within ``reach`` subtiles of A? Compared squared, so no isqrt."""
    return distance_squared(ax, ay, bx, by) <= reach * reach


def circles_overlap(
    ax: int, ay: int, radius_a: int, bx: int, by: int, radius_b: int
) -> bool:
    total = radius_a + radius_b
    return distance_squared(ax, ay, bx, by) < total * total


def point_along(
    ax: int, ay: int, bx: int, by: int, travelled: int, segment_length: int
) -> tuple[int, int]:
    """The point ``travelled`` subtiles from A toward B along the segment A->B.

    Movement is expressed this way rather than as a per-tick delta on purpose.
    Adding a truncated step vector every tick would let the truncation error
    *accumulate*: at roughly one subtile lost per tick, a unit would drift a
    fifth of a tile over a single minute. Deriving the position from a running
    total instead keeps the error bounded at one subtile no matter how long the
    unit walks.
    """
    if segment_length <= 0:
        return bx, by
    if travelled >= segment_length:
        return bx, by
    if travelled <= 0:
        return ax, ay
    return (
        ax + (bx - ax) * travelled // segment_length,
        ay + (by - ay) * travelled // segment_length,
    )


def clamp(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def ring_offsets(count: int, radius: int, *, start_eighth: int = 0) -> tuple[tuple[int, int], ...]:
    """Positions for ``count`` units spread evenly on a circle of ``radius``.

    Swarm cards do not drop their units on one point -- ``SummonRadius`` spaces
    them out (Skeletons 700, Goblin Gang 1000, Bats 750). Without this a
    Skeleton Army is fifteen units at identical coordinates: it looks like one
    unit, and every area effect hits all of them perfectly.

    A single unit lands dead centre. The trigonometry runs once here and is
    rounded to whole subtiles immediately, so no float reaches the tick loop.
    """
    if count <= 1 or radius <= 0:
        return ((0, 0),) * max(1, count)
    import math

    step = 2 * math.pi / count
    phase = start_eighth * math.pi / 4
    return tuple(
        (
            round(radius * math.cos(phase + step * i)),
            round(radius * math.sin(phase + step * i)),
        )
        for i in range(count)
    )
