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
from ..data.source import LogicData
from .constants import TickClock
from .entity import EntityKind
from .fixed import milli_tiles

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
    if hitpoints <= 0:
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
        kind = EntityKind.BUILDING if ref.startswith("BUILDING.") else EntityKind.TROOP

    speed = _int(raw.get("Speed"))
    return UnitSpec(
        name=str(raw.get("Name", name)),
        kind=kind,
        hitpoints=scale.scale(hitpoints, level),
        damage=scale.scale(damage, level) if damage else 0,
        shield_hitpoints=scale.scale(_int(raw.get("ShieldHitpoints")), level),
        hit_speed_ticks=clock.ticks(raw.get("HitSpeed")),
        load_time_ticks=clock.ticks(raw.get("LoadTime")),
        deploy_ticks=clock.ticks(raw.get("DeployTime")),
        deploy_delay_ticks=clock.ticks(raw.get("DeployDelay")),
        lifetime_ticks=clock.ticks(raw.get("LifeTime")),
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


def _projectile_damage(data: LogicData, name: Any) -> int | None:
    if not isinstance(name, str):
        return None
    try:
        projectile = data.resolve(f"PROJECTILE.{name}")
    except KeyError:
        return None
    damage = projectile.get("Damage")
    return damage if isinstance(damage, int) else None


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
