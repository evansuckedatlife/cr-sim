"""Engine-ready unit specifications.

This is the bridge between the data layer and the simulation. A resolved
attribute dict from :mod:`cr_sim.data.source` speaks the game files' units --
milliseconds, milli-tiles, tiles-per-minute, all at level 1. A :class:`UnitSpec`
speaks the engine's -- whole ticks, subtiles, subtiles-per-tick, scaled to the
level being played.

Doing the conversion once, here, is deliberate. If the hot loop had to convert
milliseconds to ticks it would do so millions of times per battle, and every
call site would be a chance to round differently and produce a unit that attacks
a tick early. Converting at build time also means an invalid spec fails loudly
when a battle is set up rather than silently misbehaving on tick 4,000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.cards import Card
from ..data.leveling import LevelTable, TowerScale
from ..data.source import LogicData, UnknownEntity
from .constants import TickClock
from .entity import EntityKind
from .fixed import SUBTILES_PER_TILE, milli_tiles

__all__ = ["UnitSpec", "build_unit_spec", "SpecError"]


class SpecError(ValueError):
    """Raised when an entity cannot be turned into a usable spec."""


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """Everything the engine needs about one unit type, pre-converted."""

    name: str
    kind: EntityKind

    # -- scaled to level, in engine units
    hitpoints: int
    damage: int
    shield_hitpoints: int

    # -- timing, in ticks
    hit_speed_ticks: int
    load_time_ticks: int
    deploy_ticks: int
    deploy_delay_ticks: int
    lifetime_ticks: int

    # -- geometry, in subtiles
    attack_range: int
    minimum_range: int
    sight_range: int
    collision_radius: int
    area_damage_radius: int

    # -- movement
    speed: int  # raw tiles/minute, kept for reporting
    speed_per_tick: int  # subtiles per tick
    mass: int
    ignore_pushback: bool

    # -- targeting
    attacks_ground: bool
    attacks_air: bool
    flying: bool
    target_only_buildings: bool
    retarget_each_tick: bool
    crown_tower_damage_percent: int

    # -- provenance, so a surprising number can be traced back
    source_ref: str
    level: int
    #: The rarity this spec was scaled with. Carried on the spec rather than
    #: looked up in a side table: anything derived from a unit later -- its
    #: projectile above all -- has to scale on the *same* ladder, and a
    #: lookup that can miss silently gives a Legendary's arrow 47% of its
    #: damage and a Champion's 39%.
    rarity: str = "Common"

    #: Consumed by its own attack (Ice Spirit, Wall Breakers, Balloon's bomb).
    kamikaze: bool = False
    #: Name of the projectile this unit launches, if it is ranged. A unit
    #: with one deals its damage on impact rather than on the swing.
    projectile: str | None = None

    #: A buff the unit puts on *itself* once it has gone this long without
    #: attacking, and loses the moment it swings again. Royal Ghost's
    #: invisibility, the Knight evolution's fortify, Suspicious Bush.
    buff_when_not_attacking: str | None = None
    buff_when_not_attacking_ticks: int = 0
    #: A buff this unit's hits inflict on whatever it hits. Electro Wizard's
    #: ZapFreeze is the important one: a stun on every hit, which is what lets
    #: him reset an Inferno Tower's ramp.
    buff_on_damage: str | None = None
    buff_on_damage_ticks: int = 0

    #: What this leaves behind when it dies. Golem's two Golemites, Lava
    #: Hound's six Pups, a hut's last defender. For the big ones this is most
    #: of what you are paying for -- a Golem that did not split would be a
    #: worse Giant, and killing one would end the push rather than start the
    #: second half of it.
    death_spawn_character: str | None = None
    #: A second, different unit spawned alongside the first (Goblin Party Hut).
    death_spawn_character2: str | None = None
    death_spawn_count: int = 0
    #: Ring radius the corpses land on. Zero puts them all on one point, which
    #: would let a single splash hit every one of them.
    death_spawn_radius: int = 0
    death_spawn_deploy_ticks: int = 0
    #: A blast on death, independent of the spawn. Golem's 88 in a 2-tile
    #: radius is why dropping a Golem on your own troops is a real cost, and
    #: Giant Skeleton's bomb is the entire card.
    death_damage: int = 0
    death_damage_radius: int = 0
    death_pushback: int = 0
    #: An area effect left where it died -- Ice Golem's slow, for instance.
    death_area_effect: str | None = None

    #: Periodic spawning, for the huts and for Witch and Tombstone. The first
    #: wave lands after ``spawn_start_ticks`` and then every
    #: ``spawn_interval_ticks``.
    spawn_character: str | None = None
    spawn_count: int = 0
    spawn_start_ticks: int = 0
    spawn_interval_ticks: int = 0
    #: How long spawning halts after the spawner is attacked or acts.
    spawn_pause_ticks: int = 0
    #: Cap on how many of its children may be alive at once.
    spawn_limit: int = 0

    #: Charge: a unit that has run far enough unobstructed hits once for
    #: ``damage_special`` instead of ``damage``, and moves at
    #: ``charge_speed_multiplier`` while winding up. Prince, Dark Prince,
    #: Battle Ram and Ram Rider all live on this.
    charge_range: int = 0
    charge_speed_multiplier: int = 0
    damage_special: int = 0

    #: Inferno ramp. Damage steps up the longer it holds one target, which is
    #: why breaking line of sight (or a stun) resets it and is the only real
    #: answer to an Inferno Tower.
    variable_damage: tuple[int, ...] = ()
    variable_damage_ticks: tuple[int, ...] = ()
    #: Inferno ramp. Damage steps up the longer it holds one target, which is
    #: why breaking line of sight (or a stun) resets it and is the only real
    #: answer to an Inferno Tower.
    variable_damage: tuple[int, ...] = ()
    variable_damage_ticks: tuple[int, ...] = ()
    #: Length of the unit's ``AttackSequence`` cycle. Monk's is 3, which is
    #: what turns his ``VariableDamage2``/``VariableDamage3`` from a *time*
    #: ladder (Inferno's, which the ticks above drive) into a *swing* ladder:
    #: 55, 55, 165 repeating, i.e. every third hit for triple. Zero for
    #: everything that has no sequence.
    attack_sequence_length: int = 0
    #: Distance the swing at each sequence index shoves its victim, in
    #: subtiles, from ``MeleePushback``/``MeleePushback2``/``MeleePushback3``.
    #: Only the third slot is ever populated in this build, and only for Monk
    #: (1800) and the event Mega Monk (3000) -- the same swing that carries the
    #: triple damage also clears the space around it.
    melee_pushback: tuple[int, ...] = ()
    #: Whether that shove reaches everything in range or only the unit hit,
    #: from ``IsMeleePushbackAll``/``...2``/``...3``.
    melee_pushback_all: tuple[bool, ...] = ()
    #: How far the *attacker* is thrown backwards by its own attack, in
    #: subtiles. Firecracker's 1.0-tile recoil is the visible one -- it is why
    #: she backs away from the bridge as she fires -- and Sparky (0.75) and the
    #: evolved Battle Ram (2.0, which is how it gets the room to charge again)
    #: carry the same field. Distinct from ``PROJECTILE.Pushback``, which is
    #: what shoves the *victim*; the two are separate columns and several
    #: cards carry one without the other.
    attack_push_back: int = 0
    #: Cannot target anything that is not a troop. Ram Rider's rider is the
    #: only reachable user: the ram charges buildings while the rider's bola
    #: only ever goes at troops.
    target_only_troops: bool = False
    #: Sparky: the windup is not restarted from zero when she retargets.
    load_first_hit: bool = False

    #: How long a unit stands still after a swing lands, on top of the normal
    #: hit-speed cooldown. Zero for every troop and building in this build --
    #: the field exists in the schema but nothing currently ships a nonzero
    #: value -- so this only matters if a future extraction populates it.
    stop_time_after_attack_ticks: int = 0

    #: A timed explosive rather than a unit: Giant Skeleton's bomb, Balloon's,
    #: Bomb Tower's. They carry no hitpoints at all -- they cannot be attacked
    #: or destroyed, they simply sit for their fuse and then go off. Modelled
    #: as an entity so the blast reuses the ordinary death payload, but flagged
    #: so nothing can target one.
    is_fuse: bool = False

    #: An ACTION graph fired when this entity appears. In this build that is
    #: where the reworked cards keep their behaviour -- the Goblin Hut's whole
    #: spawn cycle is an OnStartingAction and its stat columns are empty.
    on_starting_action: str | None = None

    #: Buffs a unit grants *itself* after landing a given number of hits, as
    #: three parallel ladders. Prince's is 2 / 4 / 6 hits for 6000 / 4000 /
    #: 2000ms of escalating rage: the longer he is left swinging, the worse he
    #: gets, which is the pressure the card is meant to apply.
    buff_after_hits: tuple[str, ...] = ()
    buff_after_hits_count: tuple[int, ...] = ()
    buff_after_hits_ticks: tuple[int, ...] = ()
    #: A buff put on whoever attacks this unit. Electro Giant's is a stun, so
    #: hitting it is itself a cost -- the card punishes the defence rather than
    #: out-damaging it.
    reflected_attack_buff: str | None = None
    reflected_attack_buff_ticks: int = 0

    @property
    def is_melee(self) -> bool:
        # 1900 milli-tiles is the game's own MELEE_RANGE_LIMIT.
        return self.attack_range <= milli_tiles(1900)

    @property
    def damage_per_second(self) -> float:
        if self.hit_speed_ticks <= 0:
            return 0.0
        return self.damage * 1000 / max(1, self.hit_speed_ticks)

    def damage_to(self, is_crown_tower: bool) -> int:
        """Damage this unit deals, reduced against towers where applicable.

        ``CrownTowerDamagePercent`` is stored as a negative percentage delta:
        Fireball's -75 means a tower takes 25% of the listed damage. Spells hit
        towers for a fraction of their damage precisely so that chip damage
        cannot substitute for pushing, and getting the sign wrong here would
        make every spell a win condition.
        """
        if not is_crown_tower or not self.crown_tower_damage_percent:
            return self.damage
        return self.damage * (100 + self.crown_tower_damage_percent) // 100


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _bool(value: Any) -> bool:
    return value is True


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _str_tuple(value: Any) -> tuple[str, ...]:
    """A field that is a list in the data, or a single value, or absent."""
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(v for v in value if isinstance(v, str) and v)
    return ()


def _int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(v for v in value if isinstance(v, int) and not isinstance(v, bool))
    return ()


#: ``ChargeRange`` is the one distance in this file that is NOT milli-tiles.
#: Read that way every charger sits at 0.25-0.30 tiles, which is meaningless as
#: a run-up distance, and the value is near-identical across units whose speed,
#: reach and damage all differ -- so it is not scaling with anything. Read as
#: hundredths of a tile it is 2.5 tiles for the Prince, which is exactly the
#: charge-up distance the card is known for, and 3.0 for Dark Prince and Battle
#: Ram. See `charge-range-unit` in reference/anchors.json.
_CHARGE_RANGE_PER_TILE = 100


def _charge_range(value: Any) -> int:
    raw = _int(value)
    return raw * SUBTILES_PER_TILE // _CHARGE_RANGE_PER_TILE if raw else 0


def _variable_damage(raw: Mapping[str, Any], scale: Any, level: int) -> tuple[int, ...]:
    """The Inferno ramp, as an ordered damage ladder.

    Stage one is the unit's ordinary ``Damage``; ``VariableDamage2`` and
    ``VariableDamage3`` are the steps above it. Inferno Tower reads 17 / 62 /
    331, which is the 20x escalation that makes it the answer to a tank and
    useless against a swarm.
    """
    steps = [_int(raw.get("Damage"))]
    for key in ("VariableDamage2", "VariableDamage3"):
        if key in raw:
            steps.append(_int(raw.get(key)))
    if len(steps) == 1:
        return ()
    return tuple(scale.scale(step, level) for step in steps)


def _variable_damage_ticks(raw: Mapping[str, Any], clock: TickClock) -> tuple[int, ...]:
    """How long each ramp stage lasts before the next one takes over."""
    times = []
    for key in ("VariableDamageTime1", "VariableDamageTime2"):
        if key in raw:
            times.append(clock.ticks(raw.get(key)))
    return tuple(times)


def build_unit_spec(
    data: LogicData,
    levels: LevelTable,
    name: str,
    *,
    level: int,
    rarity: str,
    clock: TickClock | None = None,
    kind: EntityKind | None = None,
) -> UnitSpec:
    """Resolve ``name`` and convert it into engine units at ``level``.

    ``level`` is the rarity's internal level; use
    :meth:`RarityScale.internal_level` to convert from the level the game
    displays.
    """
    clock = clock or TickClock()
    try:
        raw: Mapping[str, Any] = data.resolve(name)
    except KeyError as exc:
        raise SpecError(f"cannot resolve entity {name!r}") from exc

    scale = levels.get(rarity)
    hitpoints = _int(raw.get("Hitpoints"))
    # A bomb has no hitpoints because there is nothing to shoot. It is a fuse:
    # ``DeployTime`` is how long it sits before going off -- 3000ms for Giant
    # Skeleton's, which is the three-second delay the card is known for. Left
    # to the guard below it would raise, `_spawn_units` would swallow the
    # error, and the Giant Skeleton would die leaving nothing, deleting the
    # entire card.
    is_fuse = hitpoints <= 0 and bool(
        _int(raw.get("DeathDamage"))
        or raw.get("DeathAreaEffect")
        or raw.get("DeathSpawnCharacter")
    )
    if is_fuse:
        hitpoints = 1
    elif hitpoints <= 0:
        raise SpecError(f"{name!r} has no hitpoints; not a spawnable unit")

    damage, source = _resolve_damage(data, raw)

    # Damage sourced from a projectile is delivered by that projectile, which
    # takes time to arrive. Melee damage lands on the swing.
    projectile = None
    if source in ("CustomFirstProjectile", "Projectile"):
        projectile = raw.get(source)
    elif source == "AttackSequenceList.Projectile":
        projectile = None

    ref = str(raw.get("__ref__", name))
    if kind is None:
        # Namespace first, then the row's own ``IsBuilding`` flag. Four entries
        # in this build sit in ``CHARACTER`` and declare themselves buildings
        # anyway, and one of them is reachable from a standard card:
        # ``BrokenCannon``, what a Cannon Cart becomes at half health. Read as
        # a troop it keeps a troop's immunity to building-targeting units, so
        # a Giant walks straight past the wreck it is supposed to stop and hit.
        kind = (
            EntityKind.BUILDING
            if ref.startswith("BUILDING.") or _bool(raw.get("IsBuilding"))
            else EntityKind.TROOP
        )

    speed = _int(raw.get("Speed"))
    return UnitSpec(
        name=str(raw.get("Name", name)),
        kind=kind,
        hitpoints=scale.scale(hitpoints, level),
        damage=scale.scale(damage, level) if damage else 0,
        shield_hitpoints=scale.scale(_int(raw.get("ShieldHitpoints")), level),
        hit_speed_ticks=clock.ticks(raw.get("HitSpeed")),
        load_time_ticks=clock.ticks(raw.get("LoadTime")),
        # A fuse spends its DeployTime counting down to its own death rather
        # than waiting to become active, so the time goes to lifetime instead.
        deploy_ticks=0 if is_fuse else clock.ticks(raw.get("DeployTime")),
        deploy_delay_ticks=clock.ticks(raw.get("DeployDelay")),
        lifetime_ticks=(
            clock.ticks(raw.get("DeployTime")) if is_fuse
            else clock.ticks(raw.get("LifeTime"))
        ),
        is_fuse=is_fuse,
        on_starting_action=_opt_str(raw.get("OnStartingAction")) or None,
        buff_after_hits=_str_tuple(raw.get("BuffAfterHits")),
        buff_after_hits_count=_int_tuple(raw.get("BuffAfterHitsCount")),
        buff_after_hits_ticks=tuple(
            clock.ticks(v) for v in _int_tuple(raw.get("BuffAfterHitsTime"))
        ),
        reflected_attack_buff=_opt_str(raw.get("ReflectedAttackBuff")) or None,
        reflected_attack_buff_ticks=clock.ticks(raw.get("ReflectedAttackBuffDuration")),
        attack_range=milli_tiles(_int(raw.get("Range"))),
        minimum_range=milli_tiles(_int(raw.get("MinimumRange"))),
        sight_range=milli_tiles(_int(raw.get("SightRange"))),
        collision_radius=milli_tiles(_int(raw.get("CollisionRadius"))),
        area_damage_radius=milli_tiles(_int(raw.get("AreaDamageRadius"))),
        speed=speed,
        speed_per_tick=clock.subtiles_per_tick(speed),
        mass=_int(raw.get("Mass")),
        ignore_pushback=_bool(raw.get("IgnorePushback")),
        attacks_ground=_bool(raw.get("AttacksGround")),
        attacks_air=_bool(raw.get("AttacksAir")),
        flying=_int(raw.get("FlyingHeight")) > 0 or _bool(raw.get("Hovering")),
        target_only_buildings=_bool(raw.get("TargetOnlyBuildings")),
        retarget_each_tick=_bool(raw.get("RetargetEachTick")),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
        kamikaze=_bool(raw.get("Kamikaze")),
        projectile=projectile if isinstance(projectile, str) else None,
        buff_when_not_attacking=_opt_str(raw.get("BuffWhenNotAttacking")),
        buff_when_not_attacking_ticks=clock.ticks(raw.get("BuffWhenNotAttackingTime")),
        buff_on_damage=_opt_str(raw.get("BuffOnDamage")),
        buff_on_damage_ticks=clock.ticks(raw.get("BuffOnDamageTime")),
        death_spawn_character=_opt_str(raw.get("DeathSpawnCharacter")) or None,
        death_spawn_character2=_opt_str(raw.get("DeathSpawnCharacter2")) or None,
        death_spawn_count=_int(raw.get("DeathSpawnCount"), 1),
        death_spawn_radius=milli_tiles(_int(raw.get("DeathSpawnRadius"))),
        death_spawn_deploy_ticks=clock.ticks(raw.get("DeathSpawnDeployTime")),
        death_damage=scale.scale(_int(raw.get("DeathDamage")), level),
        death_damage_radius=milli_tiles(_int(raw.get("DeathDamageRadius"))),
        death_pushback=milli_tiles(_int(raw.get("DeathPushBack"))),
        death_area_effect=_opt_str(raw.get("DeathAreaEffect")) or None,
        spawn_character=_opt_str(raw.get("SpawnCharacter")) or None,
        spawn_count=_int(raw.get("SpawnNumber"), 1),
        spawn_start_ticks=clock.ticks(raw.get("SpawnStartTime")),
        spawn_interval_ticks=clock.ticks(raw.get("SpawnInterval")),
        spawn_pause_ticks=clock.ticks(raw.get("SpawnPauseTime")),
        spawn_limit=_int(raw.get("SpawnLimit")),
        charge_range=_charge_range(raw.get("ChargeRange")),
        charge_speed_multiplier=_int(raw.get("ChargeSpeedMultiplier")),
        damage_special=scale.scale(_int(raw.get("DamageSpecial")), level),
        variable_damage=_variable_damage(raw, scale, level),
        variable_damage_ticks=_variable_damage_ticks(raw, clock),
        load_first_hit=_bool(raw.get("LoadFirstHit")),
        stop_time_after_attack_ticks=clock.ticks(raw.get("StopTimeAfterAttack")),
        source_ref=ref,
        level=level,
        rarity=rarity,
    )


def build_tower_spec(
    data: LogicData,
    name: str,
    scale: TowerScale,
    *,
    level: int,
    clock: TickClock | None = None,
) -> UnitSpec:
    """Build a Crown Tower spec using the tower progression, not the card ladder.

    Towers accumulate a flat percentage of their level-1 base (8%/level for a
    Princess, 7% for the King, 10% past level 9) rather than compounding on the
    card multiplier table. Using the card ladder here inflates a level-11
    Princess Tower from 2576 to 3584 hitpoints, which would skew every damage
    race in the simulator.
    """
    clock = clock or TickClock()
    raw = data.resolve(name)
    hitpoints = _int(raw.get("Hitpoints"))
    if hitpoints <= 0:
        raise SpecError(f"{name!r} has no hitpoints; not a tower")
    damage, source = _resolve_damage(data, raw)
    projectile = raw.get(source) if source in ("CustomFirstProjectile", "Projectile") else None

    return UnitSpec(
        name=str(raw.get("Name", name)),
        kind=EntityKind.TOWER,
        hitpoints=scale.hitpoints(hitpoints, level),
        damage=scale.damage(damage, level) if damage else 0,
        shield_hitpoints=0,
        hit_speed_ticks=clock.ticks(raw.get("HitSpeed")),
        load_time_ticks=clock.ticks(raw.get("LoadTime")),
        deploy_ticks=0,
        deploy_delay_ticks=0,
        lifetime_ticks=0,
        attack_range=milli_tiles(_int(raw.get("Range"))),
        minimum_range=milli_tiles(_int(raw.get("MinimumRange"))),
        sight_range=milli_tiles(_int(raw.get("SightRange"))),
        collision_radius=milli_tiles(_int(raw.get("CollisionRadius"))),
        area_damage_radius=milli_tiles(_int(raw.get("AreaDamageRadius"))),
        speed=0,
        speed_per_tick=0,
        mass=_int(raw.get("Mass"), 1000),
        ignore_pushback=True,
        attacks_ground=_bool(raw.get("AttacksGround")),
        attacks_air=_bool(raw.get("AttacksAir")),
        flying=False,
        target_only_buildings=False,
        retarget_each_tick=_bool(raw.get("RetargetEachTick")),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
        projectile=projectile if isinstance(projectile, str) else None,
        stop_time_after_attack_ticks=clock.ticks(raw.get("StopTimeAfterAttack")),
        source_ref=str(raw.get("__ref__", name)),
        level=level,
        rarity="Common",
    )


def _resolve_damage(data: LogicData, raw: Mapping[str, Any]) -> tuple[int, str]:
    """Per-hit damage, following the same chain the data layer established.

    Kept in step with :func:`cr_sim.data.cards._character_damage`: melee units
    carry ``Damage`` directly, ranged ones keep it on a projectile, Princess
    hides it behind ``CustomFirstProjectile`` because her plain ``Projectile``
    is decorative, and reworked units keep per-swing values in
    ``AttackSequenceList``.
    """
    damage = raw.get("Damage")
    if isinstance(damage, int):
        return damage, "character"

    for field in ("CustomFirstProjectile", "Projectile"):
        found = _projectile_damage(data, raw.get(field))
        if found is not None:
            return found, field

    sequence = raw.get("AttackSequenceList")
    if isinstance(sequence, (list, tuple)):
        for step in sequence:
            if not isinstance(step, Mapping):
                continue
            value = step.get("Damage")
            if isinstance(value, int):
                return value, "AttackSequenceList"
            found = _projectile_damage(data, step.get("Projectile"))
            if found is not None:
                return found, "AttackSequenceList.Projectile"
    return 0, "none"


def _projectile_damage(data: LogicData, name: Any, _depth: int = 0) -> int | None:
    """Damage a named projectile delivers, following ``SpawnProjectile``.

    A projectile may be pure delivery and keep its damage on what it turns
    into. Firecracker is the clear case: ``FirecrackerProjectile`` carries no
    damage at all and spawns ``FirecrackerExplosion``, which carries 25 -- so a
    reader that stopped at the first projectile gave her a damage of zero and
    made a card that fights into one that does not.

    The same chain is already followed when building the projectile itself, for
    the Log; this keeps the unit's own damage figure in step with it.

    Bounded, because a chain is data and data can be circular.
    """
    if not isinstance(name, str) or _depth > 4:
        return None
    try:
        projectile = data.resolve(f"PROJECTILE.{name}")
    except (KeyError, UnknownEntity):
        return None
    damage = projectile.get("Damage")
    if isinstance(damage, int):
        return damage
    return _projectile_damage(data, projectile.get("SpawnProjectile"), _depth + 1)


def spec_for_card(
    data: LogicData,
    levels: LevelTable,
    card: Card,
    *,
    display_level: int = 11,
    clock: TickClock | None = None,
) -> tuple[UnitSpec, ...]:
    """Every unit a card deploys, as specs at the given displayed level."""
    scale = levels.get(card.rarity)
    level = scale.internal_level(display_level)
    specs: list[UnitSpec] = []
    for character, count in card.summons():
        spec = build_unit_spec(
            data, levels, character, level=level, rarity=card.rarity, clock=clock
        )
        specs.extend([spec] * count)
    return tuple(specs)
