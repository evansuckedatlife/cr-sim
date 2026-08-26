"""The attack cycle and damage application.

An attack in Clash Royale is not "deal damage every ``HitSpeed`` ticks". It is a
small state machine, and the details decide interactions:

``LoadTime``
    The windup before the **first** hit after engaging. A Knight waits 700ms
    before its first swing but only 1200ms between later ones. This is why a
    unit that keeps being forced to re-engage never actually deals damage, and
    why Inferno Tower resets are devastating.
``HitSpeed``
    The interval between subsequent hits once loaded.
``StopTimeAfterAttack``
    How long a unit stands still after swinging, which is why melee units
    "stutter" toward a target rather than gliding.

Damage lands **at the end** of a windup, not the start -- a unit killed during
its load deals nothing, which is the entire basis of counter-pushing.

The load is tracked as a countdown that persists across ticks rather than as a
timestamp, so it survives being paused (freeze) or reset (a new target) without
any date arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entity import Entity, EntityKind, EntityState
from .fixed import distance_squared
from .specs import UnitSpec

__all__ = [
    "AttackState",
    "DamageEvent",
    "PendingHit",
    "advance_attack",
    "apply_hit",
    "apply_area_damage",
]


@dataclass(slots=True)
class AttackState:
    """One unit's place in its attack cycle."""

    #: Ticks until the next hit lands. Zero means "ready to swing now".
    cooldown: int = 0
    #: True once the unit has landed a hit on its current target, so later hits
    #: use HitSpeed rather than the longer first-hit LoadTime.
    loaded: bool = False
    #: Ticks the unit must stand still after a swing.
    stop_ticks: int = 0
    #: The target the current load belongs to; changing target restarts it.
    target_id: int = 0

    def engage(self, spec: UnitSpec, target_id: int) -> None:
        """Begin winding up against a target.

        Switching target restarts the windup from ``LoadTime``. That is what
        makes distraction powerful: a unit pulled onto a new target has to pay
        the first-hit delay again.
        """
        if target_id != self.target_id:
            self.target_id = target_id
            self.loaded = False
            self.cooldown = max(1, spec.load_time_ticks)

    def disengage(self) -> None:
        self.target_id = 0
        self.loaded = False
        self.cooldown = 0

    @property
    def can_move(self) -> bool:
        return self.stop_ticks <= 0


@dataclass(slots=True)
class DamageEvent:
    """A hit that landed, for logging and the interaction tests."""

    tick: int
    attacker_id: int
    target_id: int
    amount: int
    lethal: bool = False


def _damage_for(spec: UnitSpec, target: Entity) -> int:
    return spec.damage_to(is_crown_tower=target.kind is EntityKind.TOWER)


@dataclass(slots=True)
class PendingHit:
    """A hit that has been *decided* this tick but not yet applied."""

    attacker: Entity
    spec: UnitSpec
    target: Entity


def advance_attack(
    state: AttackState,
    spec: UnitSpec,
    attacker: Entity,
    target: Entity,
) -> PendingHit | None:
    """Advance one unit's attack cycle by a tick, without dealing damage yet.

    Deciding and applying are deliberately separate. If damage landed inline,
    the outcome of a fight would depend on the order entities happen to sit in
    the entity list: whichever unit iterated first would land the killing blow
    and the other -- already at zero hitpoints -- would be skipped before it
    could swing. Two identical units placed symmetrically would then produce a
    winner, which is plainly wrong.

    Collecting hits and applying them together makes a tick simultaneous, so a
    mirror match trades evenly and no result depends on spawn order.
    """
    if state.stop_ticks > 0:
        state.stop_ticks -= 1

    state.engage(spec, target.id)
    attacker.set_state(EntityState.ATTACKING)

    if state.cooldown > 0:
        state.cooldown -= 1
        if state.cooldown > 0:
            return None

    # The windup finished on this tick, so this unit swings.
    state.loaded = True
    state.cooldown = max(1, spec.hit_speed_ticks)
    state.stop_ticks = 0
    return PendingHit(attacker=attacker, spec=spec, target=target)


def apply_hit(hit: PendingHit, tick: int) -> DamageEvent | None:
    """Apply a decided hit. Its target may already have died this tick."""
    dealt = hit.target.apply_damage(_damage_for(hit.spec, hit.target))
    if not dealt:
        return None
    return DamageEvent(
        tick=tick,
        attacker_id=hit.attacker.id,
        target_id=hit.target.id,
        amount=dealt,
        lethal=hit.target.hitpoints <= 0,
    )


def apply_area_damage(
    spec: UnitSpec,
    origin: tuple[int, int],
    radius: int,
    targets: list[Entity],
    attacker: Entity,
    tick: int,
) -> list[DamageEvent]:
    """Splash damage around a point.

    Area damage is measured to each victim's *centre*, unlike attack range which
    measures hitbox gaps -- a large unit is not easier to splash for being
    large, it simply occupies the radius.
    """
    events: list[DamageEvent] = []
    radius_squared = radius * radius
    for target in targets:
        if target.dead or target.team is attacker.team or not target.is_targetable:
            continue
        if target.flying and not spec.attacks_air:
            continue
        if not target.flying and not spec.attacks_ground:
            continue
        if distance_squared(origin[0], origin[1], target.x, target.y) > radius_squared:
            continue
        dealt = target.apply_damage(_damage_for(spec, target))
        if dealt:
            events.append(
                DamageEvent(
                    tick=tick,
                    attacker_id=attacker.id,
                    target_id=target.id,
                    amount=dealt,
                    lethal=target.hitpoints <= 0,
                )
            )
    return events
