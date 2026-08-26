"""Projectiles in flight.

Until now a ranged attack resolved the instant it was decided. That is fine for
a Knight's sword and near enough for a tower, and badly wrong for everything
slow: a Mortar shell takes over two seconds to land, and in that time its target
can walk out from under it. Instant resolution deletes a whole layer of the
game -- dodging, over-committing, spells landing where a push *was*.

**Speed is in tiles per minute**, the same unit as a character's, so a
projectile covers ``Speed * SUBTILES_PER_TILE / (60 * tps)`` subtiles a tick --
exactly the conversion units use. The reading is corroborated by three shots
whose flight times are recognisable in play:

============  ======  ============  ===========================
projectile    Speed   over its range  flight time
============  ======  ============  ===========================
Mortar shell     300    11.5 tiles   2300ms -- a slow visible lob
X-Bow bolt      1600    11.5 tiles    431ms -- near-continuous fire
King Tower      1000     7.0 tiles    420ms -- a quick flat shot
============  ======  ============  ===========================

**Homing versus not** is the distinction that matters mechanically. A homing
projectile follows its target and effectively cannot be dodged; a ground-aimed
one commits to the position it was fired at, which is what makes a Mortar or a
Rocket a prediction rather than a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.leveling import RarityScale
from ..data.source import LogicData, UnknownEntity
from .constants import TickClock
from .entity import Entity, EntityKind, Team
from .fixed import SUBTILES_PER_TILE, distance, milli_tiles, point_along

__all__ = ["ProjectileSpec", "build_projectile_spec", "Projectile", "flight_ticks"]


@dataclass(frozen=True, slots=True)
class ProjectileSpec:
    """A projectile type, pre-converted to engine units."""

    name: str
    speed: int  # raw tiles/minute, kept for reporting
    speed_per_tick: int  # subtiles per tick
    damage: int
    radius: int  # splash radius in subtiles; 0 = single target
    homing: bool
    aoe_to_air: bool
    aoe_to_ground: bool
    pushback: int
    crown_tower_damage_percent: int
    target_buff: str | None
    buff_ticks: int
    spawn_character: str | None
    spawn_count: int
    #: An area effect this shot leaves where it lands, if any.
    area_effect: str | None = None

    @property
    def is_splash(self) -> bool:
        return self.radius > 0

    def damage_to(self, is_crown_tower: bool) -> int:
        if not is_crown_tower or not self.crown_tower_damage_percent:
            return self.damage
        return self.damage * (100 + self.crown_tower_damage_percent) // 100


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _bool(value: Any) -> bool:
    return value is True


def flight_ticks(distance_subtiles: int, speed: int, clock: TickClock) -> int:
    """How long a shot of this speed takes to cover a distance."""
    per_tick = speed * SUBTILES_PER_TILE // (60 * clock.ticks_per_second)
    if per_tick <= 0:
        return 0
    return max(1, distance_subtiles // per_tick)


def build_projectile_spec(
    data: LogicData,
    name: str,
    scale: RarityScale,
    *,
    level: int,
    clock: TickClock | None = None,
) -> ProjectileSpec | None:
    """Resolve a projectile and scale its damage to ``level``."""
    clock = clock or TickClock()
    try:
        raw: Mapping[str, Any] = data.resolve(f"PROJECTILE.{name}")
    except (UnknownEntity, KeyError):
        return None

    speed = _int(raw.get("Speed"))
    damage = _int(raw.get("Damage"))

    # A projectile can be pure delivery, carrying no damage of its own and
    # spawning the thing that does. The Log is thrown, lands, and *becomes* a
    # rolling log; Barbarian Log does the same before dropping a Barbarian.
    # Follow the chain for the payload rather than reporting zero damage.
    #
    # NOTE: the roll itself is not simulated -- the damage is delivered where
    # the throw lands rather than swept along the lane behind it. That
    # understates the Log against a spread-out push and is recorded as a known
    # limitation rather than passed off as complete.
    spawned = raw.get("SpawnProjectile")
    if not damage and isinstance(spawned, str):
        try:
            follow: Mapping[str, Any] = data.resolve(f"PROJECTILE.{spawned}")
        except (UnknownEntity, KeyError):
            follow = {}
        damage = _int(follow.get("Damage"))
        if damage:
            raw = {**raw, **{k: v for k, v in follow.items() if k != "Name"}}
    return ProjectileSpec(
        name=str(raw.get("Name", name)),
        speed=speed,
        speed_per_tick=speed * SUBTILES_PER_TILE // (60 * clock.ticks_per_second),
        damage=scale.scale(damage, level) if damage else 0,
        radius=milli_tiles(_int(raw.get("Radius"))),
        homing=_bool(raw.get("Homing")),
        aoe_to_air=_bool(raw.get("AoeToAir")),
        aoe_to_ground=_bool(raw.get("AoeToGround")),
        pushback=milli_tiles(_int(raw.get("Pushback"))),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
        target_buff=raw.get("TargetBuff") if isinstance(raw.get("TargetBuff"), str) else None,
        buff_ticks=clock.ticks(raw.get("BuffTime")),
        spawn_character=(
            raw.get("SpawnCharacter") if isinstance(raw.get("SpawnCharacter"), str) else None
        ),
        spawn_count=_int(raw.get("SpawnCharacterCount"), 1),
        area_effect=(
            raw.get("SpawnAreaEffectObject")
            if isinstance(raw.get("SpawnAreaEffectObject"), str)
            else None
        ),
    )


class Projectile(Entity):
    """A shot in flight.

    Modelled as an entity so it lands in the state hash and the replay viewer
    for free, and excluded from targeting and collision so nothing tries to
    shoot it or walk into it.

    A non-homing shot stores the point it was aimed at and commits to it; a
    homing one re-reads its target's position every tick. That single difference
    is what separates a dodgeable Mortar shell from an unavoidable arrow.
    """

    __slots__ = (
        "pspec", "target_x", "target_y", "travelled", "total", "origin",
        "owner_id", "ticks_left",
    )

    #: Hard cap on flight time. Nothing in the game should need it -- it exists
    #: so a mistake in the homing maths can never strand a shot in the air
    #: forever, silently accumulating entities for the rest of the match.
    MAX_FLIGHT_TICKS = 600

    def __init__(
        self,
        *,
        pspec: ProjectileSpec,
        team: Team,
        x: int,
        y: int,
        target: Entity,
        owner_id: int,
        spawn_tick: int,
    ) -> None:
        super().__init__(
            kind=EntityKind.PROJECTILE,
            team=team,
            x=x,
            y=y,
            hitpoints=1,
            spawn_tick=spawn_tick,
        )
        self.pspec = pspec
        self.target_id = target.id
        self.target_x = target.x
        self.target_y = target.y
        self.origin = (x, y)
        self.travelled = 0
        self.total = max(1, distance(x, y, target.x, target.y))
        self.owner_id = owner_id
        self.ticks_left = self.MAX_FLIGHT_TICKS

    def advance(self, target: Entity | None) -> bool:
        """Fly one tick. Returns True on the tick it arrives.

        A homing shot re-aims from **where it currently is**, not from where it
        was fired. Re-measuring from the launch point instead lets a retreating
        target outrun the remaining distance forever, and the shot never lands.

        If the target dies mid-flight the shot keeps going to where it last was
        and is spent there. A projectile in the air is something the attacker
        has already committed.
        """
        self.ticks_left -= 1
        if self.ticks_left <= 0:
            return True  # expire where it is rather than fly forever
        step = self.pspec.speed_per_tick
        if self.pspec.homing and target is not None and not target.dead:
            self.target_x, self.target_y = target.x, target.y
            self.origin = (self.x, self.y)
            self.travelled = 0
            self.total = max(1, distance(self.x, self.y, self.target_x, self.target_y))
            if self.total <= step:
                self.x, self.y = self.target_x, self.target_y
                return True

        self.travelled += step
        if self.travelled >= self.total:
            self.x, self.y = self.target_x, self.target_y
            return True
        self.x, self.y = point_along(
            *self.origin, self.target_x, self.target_y, self.travelled, self.total
        )
        return False
