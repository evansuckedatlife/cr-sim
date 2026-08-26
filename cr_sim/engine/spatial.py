"""Uniform-grid spatial index.

Both of the expensive questions the engine asks each tick are neighbourhood
queries: *what can this unit see* and *what is this unit overlapping*. Answering
either by scanning every entity makes the tick O(n^2), which is exactly what
made a busy board take 23 seconds a match -- fine to verify with, useless to
train against.

A uniform grid is the right structure here rather than a quadtree or BVH:
entities are small, of similar size, spread over a fixed 18x32 board, and they
*all move every tick*. That last point is decisive -- a tree would have to be
rebuilt or repaired constantly, whereas rebuilding a flat grid of buckets is a
linear pass with no allocation churn.

The grid is rebuilt from scratch each tick rather than incrementally updated.
Incremental updates are faster in principle and a well-known source of
heisenbugs: one missed move and an entity is queryable at a position it left
minutes ago. Rebuilding cannot drift.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .entity import Entity
from .fixed import SUBTILES_PER_TILE

__all__ = ["SpatialIndex", "DEFAULT_CELL"]

#: Two tiles. Measured against one-tile cells, which lose: a smaller cell means
#: a query spans more buckets, and the per-bucket saving does not pay for the
#: extra iteration.
DEFAULT_CELL = 2 * SUBTILES_PER_TILE


class SpatialIndex:
    """Buckets entities by position for cheap radius queries."""

    __slots__ = ("cell", "columns", "rows", "_buckets", "_count", "_dirty")

    def __init__(self, width: int, height: int, cell: int = DEFAULT_CELL) -> None:
        self.cell = cell
        self.columns = max(1, width // cell + 1)
        self.rows = max(1, height // cell + 1)
        self._buckets: list[list[Entity]] = [[] for _ in range(self.columns * self.rows)]
        self._count = 0
        #: Indices of buckets left non-empty by the last rebuild -- the only
        #: ones the next rebuild needs to clear. A busy board still touches a
        #: small fraction of the grid's cells, so clearing all of them (as
        #: opposed to just the ones that were used) was pure waste: on a
        #: mid-match board this was measured as roughly a third of a tick's
        #: total cost, almost all of it clearing buckets nothing had ever put
        #: an entity into.
        self._dirty: list[int] = []

    def __deepcopy__(self, memo: dict) -> "SpatialIndex":
        """Fast path for ``copy.deepcopy`` -- what :meth:`Battle.clone` uses.

        :meth:`Battle.step` rebuilds this from ``self.entities`` before any
        phase reads it (see the top of ``step``), every single tick -- so
        whatever a branch's index held at the moment it was cloned is
        guaranteed to be overwritten before it is ever read. A fresh, empty
        grid of the same shape is exactly what a freshly-constructed
        ``Battle`` already starts with (nothing has called ``rebuild`` yet
        either), and it is what this would be replaced with regardless.
        Generic deepcopy was instead recursing into every one of
        ``columns * rows`` bucket lists -- mostly empty on any real board --
        and every entity referenced by the ones that were not, on every
        single clone.
        """
        clone = SpatialIndex.__new__(SpatialIndex)
        clone.cell = self.cell
        clone.columns = self.columns
        clone.rows = self.rows
        clone._buckets = [[] for _ in range(self.columns * self.rows)]
        clone._count = 0
        clone._dirty = []
        return clone

    def rebuild(self, entities: Iterable[Entity]) -> None:
        buckets = self._buckets
        for index in self._dirty:
            buckets[index].clear()
        dirty: list[int] = []
        count = 0
        columns, cell = self.columns, self.cell
        rows = self.rows
        for entity in entities:
            if entity.dead:
                continue
            cx = entity.x // cell
            cy = entity.y // cell
            if cx < 0:
                cx = 0
            elif cx >= columns:
                cx = columns - 1
            if cy < 0:
                cy = 0
            elif cy >= rows:
                cy = rows - 1
            index = cy * columns + cx
            bucket = buckets[index]
            if not bucket:
                dirty.append(index)
            bucket.append(entity)
            count += 1
        self._dirty = dirty
        self._count = count

    def __len__(self) -> int:
        return self._count

    def near(self, x: int, y: int, radius: int) -> Iterator[Entity]:
        """Every entity in the cells overlapping the query circle.

        This is a *broad phase*: it returns candidates, not an exact answer. The
        caller still checks real distances. Over-returning is cheap; missing a
        neighbour would be a correctness bug, so the cell span is rounded
        outward.
        """
        cell = self.cell
        columns, rows = self.columns, self.rows
        low_x = (x - radius) // cell
        high_x = (x + radius) // cell
        low_y = (y - radius) // cell
        high_y = (y + radius) // cell
        if low_x < 0:
            low_x = 0
        if low_y < 0:
            low_y = 0
        if high_x >= columns:
            high_x = columns - 1
        if high_y >= rows:
            high_y = rows - 1
        buckets = self._buckets
        for cy in range(low_y, high_y + 1):
            base = cy * columns
            for cx in range(low_x, high_x + 1):
                yield from buckets[base + cx]

    def candidates(self, entity: Entity, radius: int) -> Iterator[Entity]:
        """Neighbours of ``entity`` within ``radius``, excluding itself."""
        for other in self.near(entity.x, entity.y, radius):
            if other is not entity:
                yield other

    def pairs(self, largest_radius: int) -> Iterator[tuple[Entity, Entity]]:
        """Every potentially-overlapping pair, each yielded once.

        The search radius is per entity -- its own radius plus the largest on
        the board -- rather than one global figure. Using the global maximum for
        everybody made every Skeleton search a King-Tower-sized neighbourhood,
        which is most of the cost of a crowded tick.

        Deduplicated by entity id rather than by tracking visited pairs: an id
        comparison is cheaper and deterministic, and collision order must not
        vary between runs.
        """
        for entity in self.all():
            reach = entity.collision_radius + largest_radius
            for other in self.candidates(entity, reach):
                if entity.id < other.id:
                    yield entity, other

    def all(self) -> Iterator[Entity]:
        for bucket in self._buckets:
            yield from bucket

    def occupancy(self) -> tuple[int, int]:
        """``(non-empty cells, largest bucket)`` -- for tuning the cell size."""
        sizes = [len(b) for b in self._buckets if b]
        return len(sizes), max(sizes, default=0)
