"""Area effect objects -- the lingering half of a spell.

A spell is rarely just damage at a point. Poison is a cloud that keeps hurting
for eight seconds, Tornado drags everything toward its centre for one, Freeze
holds a whole push still for four. All of that is one mechanism in the game
data: an **area effect object** (``AEO``), a thing placed on the board with a
lifetime that repeatedly touches whatever is inside its radius.

Three fields shape everything:

``LifeDuration``
    How long it lasts. A one-shot blast like Zap has a lifetime of a single
    millisecond -- it exists for one tick, hits, and is gone. Poison lasts
    8000ms. Same mechanism, wildly different feel.
``HitSpeed``
    The interval between applications. Absent means "apply once". Tornado's is
    50ms -- twenty times a second, which is why it feels continuous.
``Buff`` / ``BuffTime``
    What it *leaves behind*. This is where the interesting spells live: Freeze
    deals almost no damage and is devastating because of the status it applies.

The distinction that matters mechanically: an area effect keeps applying to
whatever is inside it *now*, so a unit that walks out stops taking it and a
unit that walks in starts. Nothing about the effect is bound to the units
present when it was cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.leveling import RarityScale
from ..data.source import LogicData, UnknownEntity
from .constants import TickClock
from .entity import Entity, EntityKind, Team
from .fixed import milli_tiles

__all__ = ["AreaEffectSpec", "build_area_effect_spec", "AreaEffect"]


@dataclass(frozen=True, slots=True)
class AreaEffectSpec:
    """An area effect type, pre-converted to engine units."""

    name: str
    radius: int
    damage: int
    life_ticks: int
    #: Ticks between applications. Zero means it applies once and expires.
    interval_ticks: int
    buff: str | None
    buff_ticks: int
    hits_air: bool
    hits_ground: bool
    only_enemies: bool
    only_own_troops: bool
    ignore_buildings: bool
    crown_tower_damage_percent: int
    projectile: str | None
    spawn_character: str | None
    spawn_count: int
    #: Named ACTION driving it, for the effects defined entirely in the action
    #: graph (Graveyard, Vines, Clone). Recorded so they are visible rather
    #: than silently inert; the interpreter lands in M6.
    action: str | None

    @property
    def is_instant(self) -> bool:
        """One application and done, like Zap."""
        return self.interval_ticks <= 0

    def damage_to(self, is_crown_tower: bool) -> int:
        if not is_crown_tower or not self.crown_tower_damage_percent:
            return self.damage
        return self.damage * (100 + self.crown_tower_damage_percent) // 100


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _bool(value: Any) -> bool:
    return value is True


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def build_area_effect_spec(
    data: LogicData,
    name: str,
    scale: RarityScale,
    *,
    level: int,
    clock: TickClock | None = None,
) -> AreaEffectSpec | None:
    """Resolve an area effect and scale its damage to ``level``."""
    clock = clock or TickClock()
    try:
        raw: Mapping[str, Any] = data.resolve(f"AEO.{name}")
    except (UnknownEntity, KeyError):
        return None

    damage = _int(raw.get("Damage"))
    return AreaEffectSpec(
        name=str(raw.get("Name", name)),
        radius=milli_tiles(_int(raw.get("Radius"))),
        damage=scale.scale(damage, level) if damage else 0,
        life_ticks=max(1, clock.ticks(raw.get("LifeDuration"), default=1)),
        interval_ticks=clock.ticks(raw.get("HitSpeed")),
        buff=_str(raw.get("Buff")),
        buff_ticks=clock.ticks(raw.get("BuffTime")),
        # A field that is absent means "no restriction"; only an explicit false
        # excludes a layer. Defaulting these to False would make most spells
        # hit nothing at all.
        hits_air=raw.get("HitsAir", True) is not False,
        hits_ground=raw.get("HitsGround", True) is not False,
        only_enemies=_bool(raw.get("OnlyEnemies")),
        only_own_troops=_bool(raw.get("OnlyOwnTroops")),
        ignore_buildings=_bool(raw.get("IgnoreBuildings")),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
        projectile=_str(raw.get("Projectile")),
        spawn_character=_str(raw.get("SpawnCharacter")),
        spawn_count=_int(raw.get("SpawnCharacterCount"), 1),
        action=_str(raw.get("OnStartingAction")) or _str(raw.get("OnHitAction")),
    )


class AreaEffect(Entity):
    """A live area effect sitting on the board.

    An entity so it is hashed, replayed and drawn like anything else, but it is
    excluded from targeting and collision -- a cloud is not something you can
    shoot or walk into.
    """

    __slots__ = ("aspec", "ticks_left", "ticks_to_next", "owner_id", "applications")

    def __init__(
        self,
        *,
        aspec: AreaEffectSpec,
        team: Team,
        x: int,
        y: int,
        owner_id: int,
        spawn_tick: int,
    ) -> None:
        super().__init__(
            kind=EntityKind.AREA_EFFECT,
            team=team,
            x=x,
            y=y,
            hitpoints=1,
            spawn_tick=spawn_tick,
        )
        self.aspec = aspec
        self.ticks_left = aspec.life_ticks
        # Applies on the tick it lands rather than after one interval. A Zap
        # that waited would not be instant, and a Poison cloud that waited
        # would let a unit walk through the first quarter-second untouched.
        self.ticks_to_next = 0
        self.owner_id = owner_id
        self.applications = 0

    def tick(self) -> bool:
        """Advance one tick. Returns True if the effect applies this tick."""
        applies = False
        if self.ticks_to_next <= 0:
            applies = True
            self.applications += 1
            # A one-shot effect has no interval; setting it beyond its own
            # lifetime is what stops it applying twice.
            self.ticks_to_next = (
                self.aspec.interval_ticks if self.aspec.interval_ticks > 0 else self.life_guard
            )
        else:
            self.ticks_to_next -= 1

        self.ticks_left -= 1
        if self.ticks_left <= 0:
            self.kill()
        return applies

    @property
    def life_guard(self) -> int:
        return self.aspec.life_ticks + 1

    def affects(self, entity: Entity) -> bool:
        """Whether this effect touches a given entity.

        Team filtering is explicit rather than assumed: most spells are
        ``OnlyEnemies``, but Rage, Heal and Clone deliberately affect your own
        side, and a couple affect everything on the board.
        """
        if entity.dead or entity.kind in (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT):
            return False
        if not entity.is_targetable:
            return False
        same_team = entity.team is self.team
        if self.aspec.only_enemies and same_team:
            return False
        if self.aspec.only_own_troops and not same_team:
            return False
        if entity.flying and not self.aspec.hits_air:
            return False
        if not entity.flying and not self.aspec.hits_ground:
            return False
        if self.aspec.ignore_buildings and entity.kind in (
            EntityKind.BUILDING,
            EntityKind.TOWER,
        ):
            return False
        return True
