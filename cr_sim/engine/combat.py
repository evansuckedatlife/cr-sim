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

from .buffs import apply_delta
from .entity import Entity, EntityKind, EntityState
from .fixed import distance_squared
from .specs import UnitSpec

__all__ = [
    "AttackState",
    "DamageEvent",
    "PendingHit",
    "advance_attack",
    "ramp_damage",
    "apply_hit",
    "damage_for",
    "apply_area_damage",
]


def _reduce_for_tower(spec: UnitSpec, damage: int, target: Entity) -> int:
    """Apply a spec's crown-tower reduction to a damage figure it did not come from."""
    if target.kind is not EntityKind.TOWER or not spec.crown_tower_damage_percent:
        return damage
    return damage * (100 + spec.crown_tower_damage_percent) // 100


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
    #: Ticks spent locked onto the current target without interruption. This is
    #: the Inferno ramp's clock: damage steps up the longer the beam holds, and
    #: the counter is what a retarget or a stun resets. Without it an Inferno
    #: Tower is a flat 17-damage building.
    locked_ticks: int = 0
    #: Swings this unit has landed in total, which drives ``AttackSequence``.
    #: Deliberately *not* reset by a retarget: Monk's third hit is his third
    #: hit, not his third hit on one victim, so pulling him onto something else
    #: does not save you from the one that hurts.
    swings: int = 0

    def engage(self, spec: UnitSpec, target_id: int) -> None:
        """Begin winding up against a target.

        Switching target restarts the windup from ``LoadTime``. That is what
        makes distraction powerful: a unit pulled onto a new target has to pay
        the first-hit delay again.
        """
        if target_id != self.target_id:
            self.target_id = target_id
            self.locked_ticks = 0
            if spec.load_first_hit and self.loaded:
                # Sparky. ``LoadFirstHit`` says the load is paid once, for the
                # first hit, and not again on every retarget -- which is why
                # switching her onto a new target does not buy you another
                # three seconds. It is the difference between a card you answer
                # by distracting it and a card you answer by stunning it.
                self.cooldown = max(1, spec.hit_speed_ticks)
            else:
                self.loaded = False
                self.cooldown = max(1, spec.load_time_ticks)

    def disengage(self) -> None:
        self.target_id = 0
        self.loaded = False
        self.cooldown = 0
        self.locked_ticks = 0

    def reset_load(self, spec: UnitSpec) -> None:
        """Send a charging attacker back to a full windup.

        For Sparky this is the whole counterplay: her charge survives being
        distracted but not being stunned, so a Zap costs her the three seconds
        that a new target does not.
        """
        self.loaded = False
        self.cooldown = max(1, spec.load_time_ticks)

    def break_lock(self) -> None:
        """Send the ramp back to its first stage without dropping the target.

        A stun does not make an Inferno Tower forget what it was shooting, it
        makes it start the burn again -- which is exactly why a 500ms Electro
        Wizard zap answers a card that would otherwise melt any tank.
        """
        self.locked_ticks = 0

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


def damage_for(
    spec: UnitSpec,
    target: Entity,
    attacker: Entity | None = None,
    base: int | None = None,
) -> int:
    """Damage one swing carries, before the victim's own reductions.

    The attacker's buffs scale what it deals; the victim's scale what it takes
    (in :meth:`Entity.apply_damage`). Keeping the two on opposite sides of the
    hit is what lets a raged Monk hit a fortified Knight and have both modifiers
    count exactly once.
    """
    if base is None:
        damage = spec.damage_to(is_crown_tower=target.kind is EntityKind.TOWER)
    else:
        damage = _reduce_for_tower(spec, base, target)
    if attacker is not None and attacker.buffs is not None:
        dealt = attacker.buffs.damage_multiplier()
        if dealt:
            damage = apply_delta(damage, dealt)
    return damage


def ramp_damage(
    spec: UnitSpec, locked_ticks: int, sequence_index: int = 0
) -> int | None:
    """Where a ramping attacker is on its damage ladder, or None if it has none.

    Two different ladders share the ``VariableDamage`` columns, and which one a
    unit is on is decided by whether it also carries ``VariableDamageTime``.

    **Timed.** Inferno Tower reads 17 / 62 / 331 with two 2000ms steps between
    them: four seconds from tickle to melting a Golem. The escalation is the
    card -- it is why an Inferno answers a tank and is useless against a swarm,
    and why resetting it is worth a whole card.

    **Per swing.** Monk reads 55 / 55 / 165 with *no* time steps at all, and an
    ``AttackSequence`` of ``[0, 1, 2]`` instead. Walked as a timed ladder those
    steps never advance and he swings for 55 forever, which deletes the card's
    entire mechanic: every third hit is triple damage and a knockback. The
    sequence index cycles with the unit's swing count, so the ladder repeats
    rather than topping out.
    """
    ladder = spec.variable_damage
    if not ladder:
        return None
    if spec.attack_sequence_length > 0 and not spec.variable_damage_ticks:
        return ladder[min(sequence_index, len(ladder) - 1)]
    stage = 0
    elapsed = locked_ticks
    for step in spec.variable_damage_ticks:
        if elapsed < step:
            break
        elapsed -= step
        stage += 1
    return ladder[min(stage, len(ladder) - 1)]


@dataclass(slots=True)
class PendingHit:
    """A hit that has been *decided* this tick but not yet applied."""

    attacker: Entity
    spec: UnitSpec
    target: Entity
    #: Damage for this specific swing, where it is not the spec's base. Set by
    #: the Inferno ramp and by a charge connecting. Keeping it on the hit means
    #: both mechanics land through one path rather than two special cases
    #: inside the damage calculation.
    damage: int | None = None
    #: Where this swing sits in the unit's ``AttackSequence``. Carried on the
    #: hit because the knockback that rides Monk's third swing has to know it
    #: was the third, and only the attack cycle counts them.
    sequence_index: int = 0


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
    state.locked_ticks += 1
    attacker.set_state(EntityState.ATTACKING)

    if state.cooldown > 0:
        state.cooldown -= 1
        if state.cooldown > 0:
            return None

    # The windup finished on this tick, so this unit swings.
    state.loaded = True
    state.cooldown = max(1, spec.hit_speed_ticks)
    state.stop_ticks = spec.stop_time_after_attack_ticks
    index = (
        state.swings % spec.attack_sequence_length
        if spec.attack_sequence_length > 0
        else 0
    )
    state.swings += 1
    return PendingHit(
        attacker=attacker,
        spec=spec,
        target=target,
        damage=ramp_damage(spec, state.locked_ticks, index),
        sequence_index=index,
    )


def apply_hit(hit: PendingHit, tick: int) -> DamageEvent | None:
    """Apply a decided hit. Its target may already have died this tick."""
    dealt = hit.target.apply_damage(
        damage_for(hit.spec, hit.target, hit.attacker, hit.damage)
    )
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
        if target.dead or target.team is attacker.team or not target.is_acquirable:
            continue
        if target.flying and not spec.attacks_air:
            continue
        if not target.flying and not spec.attacks_ground:
            continue
        if distance_squared(origin[0], origin[1], target.x, target.y) > radius_squared:
            continue
        dealt = target.apply_damage(damage_for(spec, target))
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
