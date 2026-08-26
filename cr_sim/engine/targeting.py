"""Who a unit decides to attack.

Targeting is where most of Clash Royale's texture actually lives. A Giant
walking past a Musketeer to keep hitting a tower, a Hog Rider ignoring the
Barbarians surrounding it, a tower switching to the closest of two threats --
all of it is target selection, not damage numbers.

Three rules matter more than the rest and are easy to get subtly wrong:

**Range is measured to a hitbox, not to a point.** A unit reaches a target when
the gap between their hitboxes closes to its ``Range``, so a Giant with a 0.75
tile radius can be hit from further away than a Skeleton with 0.5. Comparing
centre-to-centre distances instead would make every large unit harder to reach
than it is.

**Targets are sticky.** A unit does not re-choose every tick. It keeps its
target until that target dies or escapes ``Range`` plus
``LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET`` -- a small grace band that stops units
flickering between two equidistant enemies. ``RetargetEachTick`` opts specific
units out.

**Building-targeting troops are not "preferring" buildings.** They cannot see
troops at all. A Giant with a Musketeer in its face has no target other than
the tower, which is why it never stops walking.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .entity import Entity, EntityKind, Team
from .fixed import distance
from .specs import UnitSpec

__all__ = [
    "can_target",
    "gap_between",
    "in_attack_range",
    "in_sight_range",
    "acquire_target",
    "should_keep_target",
    "STRUCTURE_KINDS",
    "UNTARGETABLE_KINDS",
]

#: Kinds a building-targeting troop is allowed to see.
STRUCTURE_KINDS = (EntityKind.BUILDING, EntityKind.TOWER)

#: Kinds that exist on the board but are not valid targets for anything.
UNTARGETABLE_KINDS = (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT)


def gap_between(attacker: Entity, target: Entity) -> int:
    """Distance between two entities' hitboxes, in subtiles.

    Both radii come out because ranges in Clash Royale describe the space
    between units rather than between their centres. Clamped at zero: once
    hitboxes overlap the gap cannot go negative.
    """
    centres = distance(attacker.x, attacker.y, target.x, target.y)
    gap = centres - attacker.collision_radius - target.collision_radius
    return gap if gap > 0 else 0


def can_target(spec: UnitSpec, attacker: Entity, target: Entity) -> bool:
    """Whether ``attacker`` is allowed to attack ``target`` at all.

    This is a hard filter, not a preference. A unit that fails it does not see
    the target and will walk right past it.
    """
    if target.dead or target.team is attacker.team:
        return False
    if target.kind in UNTARGETABLE_KINDS:
        # A shot in flight and a spell's cloud are both entities so they get
        # hashed, replayed and drawn -- but neither is a thing you can attack.
        # Leaving them targetable let a Knight kill a Poison cloud, which has
        # one hitpoint, and cut the spell short.
        return False
    if not target.is_acquirable:  # still deploying, or invisible
        return False
    if target.flying:
        if not spec.attacks_air:
            return False
    elif not spec.attacks_ground:
        return False
    if spec.target_only_buildings and target.kind not in STRUCTURE_KINDS:
        return False
    return True


def in_attack_range(spec: UnitSpec, attacker: Entity, target: Entity) -> bool:
    gap = gap_between(attacker, target)
    if spec.minimum_range and gap < spec.minimum_range:
        return False
    return gap <= spec.attack_range


def in_sight_range(spec: UnitSpec, attacker: Entity, target: Entity, *, bonus: int = 0) -> bool:
    return gap_between(attacker, target) <= spec.sight_range + bonus


def acquire_target(
    spec: UnitSpec,
    attacker: Entity,
    candidates: Iterable[Entity],
    *,
    sight_bonus_for_towers: int = 0,
) -> Entity | None:
    """Pick a target from ``candidates``.

    Nearest-first by hitbox gap, with entity id as a tiebreak so two units in
    identical positions never disagree -- an arbitrary but *stable* choice,
    which is what determinism requires.
    """
    best: Entity | None = None
    best_key: tuple[int, int] | None = None
    for candidate in candidates:
        if not can_target(spec, attacker, candidate):
            continue
        bonus = sight_bonus_for_towers if candidate.kind is EntityKind.TOWER else 0
        gap = gap_between(attacker, candidate)
        if gap > spec.sight_range + bonus:
            continue
        key = (gap, candidate.id)
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    return best


def should_keep_target(
    spec: UnitSpec,
    attacker: Entity,
    target: Entity | None,
    *,
    range_extension: int = 0,
) -> bool:
    """Whether a unit holds its current target for another tick.

    The extension band is what stops a unit oscillating between two enemies at
    identical distance: having committed, it needs the target to get
    meaningfully further away before letting go, not merely a subtile further
    than someone else.
    """
    if target is None or target.dead or not target.is_acquirable:
        return False
    if spec.retarget_each_tick:
        return False
    if not can_target(spec, attacker, target):
        return False
    return gap_between(attacker, target) <= spec.sight_range + range_extension


def nearest_structure(
    attacker: Entity, structures: Sequence[Entity], team: Team
) -> Entity | None:
    """The closest living enemy structure, for units with nothing else to do."""
    best: Entity | None = None
    best_key: tuple[int, int] | None = None
    for structure in structures:
        if structure.dead or structure.team is team:
            continue
        key = (distance(attacker.x, attacker.y, structure.x, structure.y), structure.id)
        if best_key is None or key < best_key:
            best, best_key = structure, key
    return best
