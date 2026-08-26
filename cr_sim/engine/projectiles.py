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

__all__ = [
    "ProjectileSpec",
    "build_projectile_spec",
    "Projectile",
    "RollingProjectile",
    "flight_ticks",
]


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

    #: How far this shot *rolls* after it lands, in subtiles. The Log and
    #: Barbarian Barrel are thrown to a point and then travel on from it,
    #: sweeping everything in a lane. Zero for everything that simply detonates.
    roll_range: int = 0
    #: Half-extents of the rolling hit area. The Log's are 1.95 tiles across
    #: and 0.6 deep -- wide and thin, which is the shape of a log lying on its
    #: side, and the reason it clears a spread-out line of troops that a round
    #: splash radius would miss.
    roll_radius_x: int = 0
    roll_radius_y: int = 0
    #: Pushback applies to everything the roll touches, not only what it kills.
    pushback_all: bool = False

    @property
    def rolls(self) -> bool:
        return self.roll_range > 0

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
    # The roll itself is modelled: see :class:`RollingProjectile`, which the
    # throw hands off to on impact.
    # The rolling stage carries its own geometry, which is nothing like the
    # throw's: the throw has a splash radius, the roll has a range and a lane.
    roll: Mapping[str, Any] = {}
    spawned = raw.get("SpawnProjectile")
    if isinstance(spawned, str):
        try:
            roll = data.resolve(f"PROJECTILE.{spawned}")
        except (UnknownEntity, KeyError):
            roll = {}

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
        roll_range=milli_tiles(_int(roll.get("ProjectileRange"))),
        roll_radius_x=milli_tiles(_int(roll.get("ProjectileRadius"))),
        roll_radius_y=milli_tiles(_int(roll.get("ProjectileRadiusY"))),
        pushback_all=_bool(roll.get("PushbackAll")),
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


class RollingProjectile(Entity):
    """A shot that keeps going after it lands, sweeping a lane.

    The Log's damage is not delivered where it is thrown -- the throw only sets
    where the roll *starts*. It then travels ``ProjectileRange`` (10.1 tiles for
    the Log, 4.5 for Barbarian Barrel) at its own speed, hitting each enemy it
    passes over exactly once and shoving them back.

    Treating it as a point detonation, as this engine did until now,
    consistently understates it: the whole reason to play a Log into a spread
    push is that it catches units the throw never lands near.

    The hit area is an ellipse rather than a circle because the data gives two
    radii -- 1.95 tiles across against 0.6 deep. That is a log lying on its
    side, and it is why the card clears a line of Goblins but not a column.
    """

    __slots__ = ("pspec", "owner_id", "direction", "travelled", "struck")

    def __init__(
        self,
        *,
        pspec: ProjectileSpec,
        team: Team,
        x: int,
        y: int,
        direction: int,
        owner_id: int,
        spawn_tick: int,
    ) -> None:
        super().__init__(
            kind=EntityKind.PROJECTILE, team=team, x=x, y=y, hitpoints=1, spawn_tick=spawn_tick
        )
        self.pspec = pspec
        self.owner_id = owner_id
        #: +1 rolls up the board, -1 down it. A log rolls away from whoever
        #: threw it; it never comes back toward its own side.
        self.direction = direction
        self.travelled = 0
        #: Each enemy is hit once. Without this the roll would tick damage into
        #: anything it moved slowly past, turning a 105-damage spell into a
        #: multi-second grinder.
        self.struck: set[int] = set()

    def advance(self, _target: Entity | None = None) -> bool:
        """Roll one tick. Returns True when it has run out of range."""
        step = self.pspec.speed_per_tick
        self.travelled += step
        self.y += step * self.direction
        return self.travelled >= self.pspec.roll_range

    def covers(self, entity: Entity) -> bool:
        """Whether the rolling area currently overlaps ``entity``.

        Compared as an ellipse, cross-multiplied so it stays in integers like
        every other distance test in this engine.
        """
        bx = self.pspec.roll_radius_x + entity.collision_radius
        by = self.pspec.roll_radius_y + entity.collision_radius
        if bx <= 0 or by <= 0:
            return False
        dx = entity.x - self.x
        dy = entity.y - self.y
        return dx * dx * by * by + dy * dy * bx * bx <= bx * bx * by * by
