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

from .arena import Arena
from .entity import Entity, EntityKind
from .fixed import distance
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


def _effective_mass(entity: Entity) -> int:
    """How strongly an entity resists being displaced."""
    if entity.kind in (EntityKind.BUILDING, EntityKind.TOWER):
        return IMMOVABLE_MASS
    spec = entity.spec
    if spec is not None and spec.ignore_pushback:
        return IMMOVABLE_MASS
    return max(1, entity.mass)


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
    gap = distance(a.x, a.y, b.x, b.y)
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
    """
    moved = 0
    for _ in range(passes):
        touched = 0
        for a, b in index.pairs(max_radius):
            # Cheap rejects first, cheapest last-to-fail ordering. The broad
            # phase returns everything in the neighbouring cells, and on a
            # contested board the large majority of those pairs are not
            # actually touching -- doing a square root on each one was the
            # single most expensive thing in the tick.
            if a.dead or b.dead:
                continue
            if a.kind in _INCORPOREAL or b.kind in _INCORPOREAL:
                continue  # shots and clouds pass over everything
            if a.flying != b.flying:
                # Air and ground occupy different layers and never collide.
                continue
            if a.deploy_ticks_left > 0 or b.deploy_ticks_left > 0:
                # A deploying unit has no presence yet: it cannot shove and
                # cannot be shoved.
                continue
            reach = a.collision_radius + b.collision_radius
            if reach <= 0:
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            if dx * dx + dy * dy >= reach * reach:
                continue  # not overlapping; no square root needed
            if separate(a, b, arena):
                touched += 1
        moved += touched
        if not touched:
            break
    return moved
