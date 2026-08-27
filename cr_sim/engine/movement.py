"""Collision, pushback, and keeping units out of terrain.

Without this a Skeleton Army is fifteen units sharing one point. Collision is
what turns a card into an *area* of board control -- why a Giant walks through
a crowd while Skeletons scatter around it, why surrounding a Prince works, why
a swarm spreads out along a bridge instead of filing through it single file.

Three rules, taken from the fields the game actually ships:

**Mass decides who moves.** ``Mass`` runs 1 (Skeleton, Bat) to 28. In a
collision each unit yields in inverse proportion to its mass, so a Skeleton
bounces off a Golem and the Golem barely notices.

**``IgnorePushback`` means immovable, not heavy.** Thirty entities set it --
Giant, Golem, P.E.K.K.A, Prince, Mega Knight, the tanks and chargers. They are
not merely hard to push; they are never displaced at all, which is why a
committed Prince charge cannot be body-blocked off its line.

**Buildings do not move and cannot be entered.** They are resolved as static
obstacles, so a unit pushed against one stops rather than sinking into it.

Overlaps are resolved *after* movement rather than by preventing it. Blocking
movement on contact would deadlock a crowd -- everyone waiting for space that
only appears once somebody moves. Letting units overlap and then separating
them always converges.
"""

from __future__ import annotations

from math import isqrt as _isqrt

from .arena import Arena
from .entity import Entity, EntityKind
from .spatial import SpatialIndex

__all__ = ["resolve_collisions", "separate", "IMMOVABLE_MASS"]

#: Effective mass for anything that cannot be pushed. Large enough that the
#: inverse-mass split gives the other party essentially all of the movement.
IMMOVABLE_MASS = 1_000_000

#: Units may not overlap by more than this before separation is forced; a small
#: tolerance stops jitter between units that are merely touching.
_TOUCH_TOLERANCE = 60  # subtiles, ~1/300 tile

#: Entity kinds with no physical presence: nothing collides with them.
_INCORPOREAL = (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT)


#: Entity kinds that never move: they are resolved as static obstacles. Named
#: at module level rather than written inline, because an inline tuple of enum
#: members is rebuilt on every call -- and this is called twice per overlapping
#: pair per relaxation pass.
_STATIC_KINDS = (EntityKind.BUILDING, EntityKind.TOWER)


def _effective_mass(entity: Entity) -> int:
    """How strongly an entity resists being displaced."""
    if entity.kind in _STATIC_KINDS:
        return IMMOVABLE_MASS
    spec = entity.spec
    if spec is not None and spec.ignore_pushback:
        return IMMOVABLE_MASS
    mass = entity.mass
    return mass if mass > 1 else 1


def separate(a: Entity, b: Entity, arena: Arena | None = None) -> bool:
    """Push two overlapping entities apart. Returns True if anything moved.

    Displacement splits by inverse mass, so the lighter unit gives way. Two
    units at the *exact* same point have no direction to separate along, so they
    are nudged apart on a fixed axis chosen by entity id -- arbitrary, but
    deterministic, which is what matters. Swarms spawn on a ring precisely so
    this case is rare.
    """
    overlap_limit = a.collision_radius + b.collision_radius
    if overlap_limit <= 0:
        return False

    dx = b.x - a.x
    dy = b.y - a.y
    # The same integer square root :func:`~cr_sim.engine.fixed.distance` takes,
    # over the separation this function has already had to compute. Calling out
    # for it would recompute both differences.
    gap = _isqrt(dx * dx + dy * dy)
    overlap = overlap_limit - gap
    if overlap <= _TOUCH_TOLERANCE:
        return False

    if gap == 0:
        # Coincident. Separate along x, ordered by id so both runs agree.
        dx, dy, gap = (overlap_limit, 0, overlap_limit) if a.id < b.id else (-overlap_limit, 0, overlap_limit)

    mass_a = _effective_mass(a)
    mass_b = _effective_mass(b)
    total = mass_a + mass_b
    if total >= 2 * IMMOVABLE_MASS:
        return False  # two immovables; nothing to do

    # Each yields in inverse proportion to its own mass.
    share_a = overlap * mass_b // total
    share_b = overlap - share_a

    moved = False
    if mass_a < IMMOVABLE_MASS and share_a:
        moved |= _shift(a, -dx * share_a // gap, -dy * share_a // gap, arena)
    if mass_b < IMMOVABLE_MASS and share_b:
        moved |= _shift(b, dx * share_b // gap, dy * share_b // gap, arena)
    return moved


def _shift(entity: Entity, dx: int, dy: int, arena: Arena | None) -> bool:
    """Move an entity, refusing to put it somewhere it cannot stand."""
    if not dx and not dy:
        return False
    x = entity.x + dx
    y = entity.y + dy
    if arena is not None and not arena.is_walkable(x, y, flying=entity.flying):
        # Try each axis alone so a unit slides along an obstacle rather than
        # sticking to it -- otherwise crowds jam solid against the river bank.
        if arena.is_walkable(entity.x + dx, entity.y, flying=entity.flying):
            entity.x += dx
            return True
        if arena.is_walkable(entity.x, entity.y + dy, flying=entity.flying):
            entity.y += dy
            return True
        return False
    entity.x = x
    entity.y = y
    return True


def resolve_collisions(
    index: SpatialIndex,
    arena: Arena | None = None,
    *,
    max_radius: int,
    passes: int = 3,
) -> int:
    """Separate every overlapping pair on the board.

    Runs a small fixed number of relaxation passes rather than iterating to
    convergence. In a dense crowd separating one pair creates another overlap,
    so true convergence could take unboundedly long and would make the cost of a
    tick depend on how crowded the board is -- exactly the property a simulator
    meant for training cannot afford.

    A residue therefore survives each tick. It is bounded, not growing: fifteen
    skeletons all walking at the same tower settle at roughly a tenth of a tile
    of mutual compression and stay there. Measured across pass counts, 2 passes
    leaves 0.23 tiles, 3 leaves 0.11, 4 leaves 0.04 for about a quarter more
    work; 3 is the knee. A crowd converging on one point compresses slightly in
    the real game too, so this is not a artefact worth paying to eliminate.

    Structurally this is :meth:`SpatialIndex.pairs` with the rejects folded
    into its innermost loop rather than applied to what it yields. The broad
    phase produces roughly ten times as many pairs as are actually touching, so
    handing every one of them back as a tuple through a generator was most of
    the phase's cost -- the pair that gets rejected never needs to exist as an
    object. Folding it in also lifts the tests that depend only on the left
    entity out of the inner loop, where they were being re-run once per
    neighbour. The two walks are pinned together by
    ``test_the_collision_sweep_matches_the_obvious_implementation``.

    Passes after the first do not re-examine the whole board. A pair whose two
    entities have both stood still since the previous pass looked at it must
    reach the same verdict it reached then -- either it was not overlapping, or
    it was and :func:`separate` declined to move anything (two immovables, or an
    overlap inside the touch tolerance). Neither outcome changes any state, so
    re-testing it is provably wasted work. What is *not* safe is the converse
    shortcut, skipping a pair because it was separated already; that one is
    exactly the pair whose geometry changed.

    The set of entities that have moved recently enough to matter is tracked as
    a bounding box in grid cells rather than as a set of entities, so the test
    an outer entity has to pass is four integer comparisons against the span it
    was going to search anyway. A box is conservative -- it can cover cells
    holding nothing that moved -- and conservative is the safe direction here:
    examining a pair that did not need examining costs time and changes
    nothing. Measured on a mid-match board, the second and third passes were
    carrying about two thirds of all the candidate pairs while only three or
    four entities had actually moved.
    """
    buckets = index._buckets
    cell = index.cell
    columns, rows = index.columns, index.rows
    moved = 0
    # Seeded to the whole grid: the first pass of a tick is the first look at
    # every pair since movement ran, so all of them are stale.
    active_low_x, active_high_x = 0, columns - 1
    active_low_y, active_high_y = 0, rows - 1
    for _ in range(passes):
        touched = 0
        # Movers of *this* pass, which is what the next pass starts from. The
        # box in force during this pass keeps growing as separations happen,
        # because an entity shoved earlier in this same pass makes its own
        # pairs stale for the rest of it.
        next_low_x, next_high_x = columns, -1
        next_low_y, next_high_y = rows, -1
        for source_index, source in enumerate(buckets):
            if not source:
                continue
            for a in source:
                # Everything here is a property of `a` alone, so it is settled
                # once rather than once per neighbour. None of it can change
                # while this pass runs: separation moves entities, it does not
                # kill them or take them out of deployment.
                if a.dead or a.deploy_ticks_left > 0:
                    continue
                if a.kind in _INCORPOREAL:
                    continue  # shots and clouds pass over everything
                radius_a = a.collision_radius
                identity = a.id
                flying = a.flying
                # The cell span is fixed from where `a` stands as the sweep
                # reaches it, exactly as the broad phase binds it when its
                # query is first stepped. The overlap test below must *not* be
                # -- an earlier separation in this same inner loop can already
                # have shifted `a` -- but the only thing that can move it is a
                # successful `separate`, so the position is cached here and
                # re-read there rather than fetched for every neighbour.
                ax = a.x
                ay = a.y
                reach = radius_a + max_radius
                low_x = (ax - reach) // cell
                high_x = (ax + reach) // cell
                low_y = (ay - reach) // cell
                high_y = (ay + reach) // cell
                if low_x < 0:
                    low_x = 0
                if low_y < 0:
                    low_y = 0
                if high_x >= columns:
                    high_x = columns - 1
                if high_y >= rows:
                    high_y = rows - 1
                # Nothing that has moved since the previous pass can be in this
                # entity's search span, so every pair it would test is one the
                # previous pass has already settled.
                if (
                    high_x < active_low_x
                    or low_x > active_high_x
                    or high_y < active_low_y
                    or low_y > active_high_y
                ):
                    continue
                for cy in range(low_y, high_y + 1):
                    base = cy * columns
                    for cx in range(low_x, high_x + 1):
                        for b in buckets[base + cx]:
                            # Cheapest rejects first. The id comparison is what
                            # makes each pair come up exactly once, and it
                            # subsumes `b is not a`.
                            if identity >= b.id:
                                continue
                            if b.flying != flying:
                                # Air and ground are different layers.
                                continue
                            # The threshold is the touching distance *less the
                            # tolerance*, not the touching distance itself:
                            # `separate` declines anything shallower than the
                            # tolerance and returns False, so testing for it
                            # here reaches the same answer without the call.
                            # A limit at or below zero means the two can never
                            # overlap deeply enough to be worth separating.
                            limit = radius_a + b.collision_radius - _TOUCH_TOLERANCE
                            if limit <= 0:
                                continue
                            dx = b.x - ax
                            dy = b.y - ay
                            if dx * dx + dy * dy >= limit * limit:
                                continue  # not overlapping; no square root
                            if b.dead or b.deploy_ticks_left > 0:
                                # A deploying unit has no presence yet: it
                                # cannot shove and cannot be shoved.
                                continue
                            if b.kind in _INCORPOREAL:
                                continue
                            if separate(a, b, arena):
                                touched += 1
                                ax = a.x  # shoved; the cache is stale
                                ay = a.y
                                # Both ends of a separated pair have moved, so
                                # both their cells join the box: the one being
                                # swept and the one the neighbour sits in. An
                                # entity's cell is its bucket, which does not
                                # change during the sweep -- the index is only
                                # rebuilt between phases.
                                source_cy, source_cx = divmod(source_index, columns)
                                low = source_cx if source_cx < cx else cx
                                high = cx if source_cx < cx else source_cx
                                if low < next_low_x:
                                    next_low_x = low
                                if high > next_high_x:
                                    next_high_x = high
                                low = source_cy if source_cy < cy else cy
                                high = cy if source_cy < cy else source_cy
                                if low < next_low_y:
                                    next_low_y = low
                                if high > next_high_y:
                                    next_high_y = high
                                # Also into the box in force for the rest of
                                # *this* pass: whatever has just been shoved
                                # makes its own pairs stale immediately.
                                if next_low_x < active_low_x:
                                    active_low_x = next_low_x
                                if next_high_x > active_high_x:
                                    active_high_x = next_high_x
                                if next_low_y < active_low_y:
                                    active_low_y = next_low_y
                                if next_high_y > active_high_y:
                                    active_high_y = next_high_y
        moved += touched
        if not touched:
            break
        active_low_x, active_high_x = next_low_x, next_high_x
        active_low_y, active_high_y = next_low_y, next_high_y
    return moved
