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
        #: ``(cx, cy, bucket)`` for every bucket left non-empty by the last
        #: rebuild -- the only ones the next rebuild needs to clear. A busy
        #: board still touches a small fraction of the grid's cells, so
        #: clearing all of them (as opposed to just the ones that were used)
        #: was pure waste: on a mid-match board this was measured as roughly a
        #: third of a tick's total cost, almost all of it clearing buckets
        #: nothing had ever put an entity into.
        #:
        #: The cell coordinates ride along because this doubles as the list of
        #: *occupied* cells, which is what :meth:`in_reach` walks when a query
        #: circle is wide enough that scanning the whole board's occupancy
        #: beats visiting every cell the circle covers.
        self._dirty: list[tuple[int, int, list[Entity]]] = []

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
        for _, _, bucket in self._dirty:
            bucket.clear()
        dirty: list[tuple[int, int, list[Entity]]] = []
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
            bucket = buckets[cy * columns + cx]
            if not bucket:
                dirty.append((cx, cy, bucket))
            bucket.append(entity)
            count += 1
        self._dirty = dirty
        self._count = count

    def __len__(self) -> int:
        return self._count

    def near(self, x: int, y: int, radius: int) -> list[Entity]:
        """Every entity in the cells overlapping the query circle.

        This is a *broad phase*: it returns candidates, not an exact answer. The
        caller still checks real distances. Over-returning is cheap; missing a
        neighbour would be a correctness bug, so the cell span is rounded
        outward.

        Returns a fresh list rather than a generator. Every caller consumes the
        whole thing, and several already wrapped it in ``list(...)`` because
        they mutate the board while iterating; ``list.extend`` copies a bucket
        at C speed, whereas a generator paid a frame resumption for every
        entity it handed back -- and it was the single hottest function in the
        tick.
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
        found: list[Entity] = []
        for cy in range(low_y, high_y + 1):
            base = cy * columns
            for cx in range(low_x, high_x + 1):
                bucket = buckets[base + cx]
                if bucket:
                    found.extend(bucket)
        return found

    def candidates(self, entity: Entity, radius: int) -> list[Entity]:
        """Neighbours of ``entity`` within ``radius``, excluding itself.

        The entity is dropped by identity from the list :meth:`near` built,
        which visits it at most once, rather than by filtering every element
        through a comparison.
        """
        found = self.near(entity.x, entity.y, radius)
        for position, other in enumerate(found):
            if other is entity:
                del found[position]
                break
        return found

    def in_reach(self, entity: Entity, radius: int) -> list[Entity]:
        """The same *set* as :meth:`candidates`, in whatever order is cheapest.

        Sight ranges are large -- a Musketeer sees 6.5 tiles, a tower further
        still -- so a targeting query spans forty to eighty of this grid's
        cells while a mid-match board only ever occupies about twenty of them.
        Visiting every cell the circle covers therefore does most of its work
        on empty ones. Walking the occupied cells instead and rejecting the
        ones outside the circle is the same answer for a fraction of the
        iterations, and the crossover between the two is just a size
        comparison, so both are kept and the cheaper is chosen per query.

        The cost is that the result comes back in occupancy order rather than
        row-major order, which is why this is a separate method rather than a
        change to :meth:`candidates`. Only a caller whose answer does not
        depend on the order may use it. Target selection qualifies and says so;
        anything that *applies* something to what it finds -- splash damage, a
        rolling log, a death nova -- does not, because the order it hits things
        in is part of the result.
        """
        cell = self.cell
        columns, rows = self.columns, self.rows
        x, y = entity.x, entity.y
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
        found: list[Entity] = []
        dirty = self._dirty
        if len(dirty) < (high_x - low_x + 1) * (high_y - low_y + 1):
            for cx, cy, bucket in dirty:
                if low_x <= cx <= high_x and low_y <= cy <= high_y:
                    found.extend(bucket)
        else:
            buckets = self._buckets
            for cy in range(low_y, high_y + 1):
                base = cy * columns
                for cx in range(low_x, high_x + 1):
                    bucket = buckets[base + cx]
                    if bucket:
                        found.extend(bucket)
        for position, other in enumerate(found):
            if other is entity:
                del found[position]
                break
        return found

    def pairs(self, largest_radius: int) -> Iterator[tuple[Entity, Entity]]:
        """Every potentially-overlapping pair, each yielded once.

        The search radius is per entity -- its own radius plus the largest on
        the board -- rather than one global figure. Using the global maximum for
        everybody made every Skeleton search a King-Tower-sized neighbourhood,
        which is most of the cost of a crowded tick.

        Deduplicated by entity id rather than by tracking visited pairs: an id
        comparison is cheaper and deterministic, and collision order must not
        vary between runs. The ``is not entity`` test :meth:`candidates` would
        apply is subsumed by it -- an id is never less than itself.

        The bucket walk is written out here rather than delegated to
        :meth:`candidates`, which delegates to :meth:`near`: three stacked
        generators cost three frame resumptions per candidate, and this is the
        highest-count loop in the engine.
        :func:`cr_sim.engine.movement.resolve_collisions` runs the same walk
        with its own filters folded into the inner loop -- the two must
        agree, and that is pinned by
        ``test_the_collision_sweep_matches_the_obvious_implementation``.
        """
        buckets = self._buckets
        cell = self.cell
        columns, rows = self.columns, self.rows
        for source in buckets:
            if not source:
                continue
            for entity in source:
                reach = entity.collision_radius + largest_radius
                identity = entity.id
                low_x = (entity.x - reach) // cell
                high_x = (entity.x + reach) // cell
                low_y = (entity.y - reach) // cell
                high_y = (entity.y + reach) // cell
                if low_x < 0:
                    low_x = 0
                if low_y < 0:
                    low_y = 0
                if high_x >= columns:
                    high_x = columns - 1
                if high_y >= rows:
                    high_y = rows - 1
                for cy in range(low_y, high_y + 1):
                    base = cy * columns
                    for cx in range(low_x, high_x + 1):
                        for other in buckets[base + cx]:
                            if identity < other.id:
                                yield entity, other

    def all(self) -> Iterator[Entity]:
        for bucket in self._buckets:
            yield from bucket

    def occupancy(self) -> tuple[int, int]:
        """``(non-empty cells, largest bucket)`` -- for tuning the cell size."""
        sizes = [len(b) for b in self._buckets if b]
        return len(sizes), max(sizes, default=0)
