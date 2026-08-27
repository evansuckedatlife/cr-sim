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
    #: Strike the largest targets rather than everything in range. Lightning
    #: uses this: three bolts at the three biggest things, which is why it
    #: answers a tank-plus-support push and not a swarm.
    hit_biggest: bool
    spawn_character: str | None
    spawn_count: int
    #: Graveyard's trickle. The card is not a burst of skeletons but a stream:
    #: one every ``SpawnInterval`` from ``SpawnInitialDelay`` until the effect
    #: expires, which is why it has to be answered over time rather than
    #: swatted once.
    spawn_initial_delay_ticks: int
    spawn_interval_ticks: int
    #: Skeletons appear in an annulus, not at the centre -- 3 to 4 tiles out for
    #: Graveyard. They surround a tower rather than piling on one tile.
    spawn_min_radius: int
    spawn_max_radius: int
    spawn_deploy_ticks: int
    spawn_randomize: bool

    @property
    def spawns_over_time(self) -> bool:
        return bool(self.spawn_character and self.spawn_interval_ticks > 0)
    #: Action fired once, where the effect lands. Graveyard's whole trickle.
    #: Either a name to look up or the action written out inline -- see
    #: :func:`_action`.
    on_start_action: "str | Mapping[str, Any] | None"
    #: Action fired **per affected entity**, every time the effect applies.
    #: Clone's is here: the spell does not do something at a point, it does
    #: something to each friendly troop it touches, and running it once at the
    #: centre would duplicate nothing.
    on_hit_action: "str | Mapping[str, Any] | None"

    @property
    def is_instant(self) -> bool:
        """One application and done, like Zap."""
        return self.interval_ticks <= 0

    @property
    def max_applications(self) -> int:
        """How many times this effect fires over its life.

        ``LifeDuration // HitSpeed``, not however many a timer happens to fit.
        The two differ for exactly one effect in the build, and it is the one
        with a published answer: Lightning is 1500ms over a 460ms interval, and
        a timer started on contact fits four bolts into that window where the
        card fires three. Every other repeating effect -- Poison at 32, Tornado
        at 21, Earthquake at 30 -- comes out identical either way, so this
        costs nothing anywhere else.

        See `lightning-bolt-count` in reference/anchors.json.
        """
        if self.interval_ticks <= 0:
            return 1
        return max(1, self.life_ticks // self.interval_ticks)

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


def _action(value: Any) -> "str | Mapping[str, Any] | None":
    """An action field: a name to look up, or the action written out in place.

    Both spellings are in the build and they are not interchangeable. Most
    effects reference an ``ACTION`` by name (``Graveyard_rework_Group``), but
    two write the whole graph inline instead: ``AEO.DarkMagicAOE`` and
    ``AEO.dead_goblinstein`` both carry an ``ActionGroup`` dict under
    ``OnStartingAction``.

    Read through a string-only helper the dict is silently discarded, and Dark
    Magic -- whose *entire* effect is that inline graph, since the row itself
    declares ``HitsAir: false`` and ``HitsGround: false`` and so touches
    nothing on its own -- becomes a 5-elixir spell that does nothing at all.
    :meth:`cr_sim.engine.actions.ActionInterpreter.run` has always accepted an
    inline row; nothing was ever reaching it.
    """
    if isinstance(value, str):
        return value or None
    if isinstance(value, Mapping):
        return value
    return None


def _layers(raw: Mapping[str, Any]) -> dict[str, bool]:
    """Which layers an area effect touches.

    The absent key is meaningful, not permissive. Across the build 115 area
    effects declare both ``HitsAir`` and ``HitsGround``, **none** declares air
    without ground, and four declare ground *without* air -- one of which is
    Earthquake, a spell whose defining property is that it does not hit air.
    So an omitted layer is an excluded layer.

    The exception is an effect declaring neither, which is an orchestrator with
    no hit-test of its own (Graveyard, Vines); those carry no damage or buff, so
    a permissive default costs nothing and avoids special-casing them.
    """
    has_air = "HitsAir" in raw
    has_ground = "HitsGround" in raw
    if not has_air and not has_ground:
        return {"hits_air": True, "hits_ground": True}
    return {
        "hits_air": raw.get("HitsAir") is True,
        "hits_ground": raw.get("HitsGround") is True,
    }


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
        **_layers(raw),
        only_enemies=_bool(raw.get("OnlyEnemies")),
        only_own_troops=_bool(raw.get("OnlyOwnTroops")),
        ignore_buildings=_bool(raw.get("IgnoreBuildings")),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
        projectile=_str(raw.get("Projectile")),
        hit_biggest=_bool(raw.get("HitBiggestTargets")),
        spawn_character=_str(raw.get("SpawnCharacter")),
        spawn_count=_int(raw.get("SpawnCharacterCount"), 1),
        spawn_initial_delay_ticks=clock.ticks(raw.get("SpawnInitialDelay")),
        spawn_interval_ticks=clock.ticks(raw.get("SpawnInterval")),
        spawn_min_radius=milli_tiles(_int(raw.get("SpawnMinRadius"))),
        spawn_max_radius=milli_tiles(
            _int(raw.get("SpawnMaxRadius")) or _int(raw.get("Radius"))
        ),
        spawn_deploy_ticks=clock.ticks(raw.get("SpawnTime")),
        spawn_randomize=_bool(raw.get("SpawnRandomizeSequence")),
        on_start_action=_action(raw.get("OnStartingAction")),
        on_hit_action=_action(raw.get("OnHitAction")),
    )


class AreaEffect(Entity):
    """A live area effect sitting on the board.

    An entity so it is hashed, replayed and drawn like anything else, but it is
    excluded from targeting and collision -- a cloud is not something you can
    shoot or walk into.
    """

    __slots__ = (
        "aspec", "ticks_left", "ticks_to_next", "owner_id", "applications", "struck",
        "ticks_to_next_spawn", "spawned",
    )

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
        #: Targets this effect has already fired a projectile at. Lightning's
        #: bolts go to different units -- striking the same one repeatedly
        #: would make it a single-target nuke rather than the answer to a tank
        #: and its support.
        self.struck: set[int] = set()
        #: Graveyard's first skeleton is late on purpose -- 2200ms after the
        #: cast, which is the window a defender has to answer before any of it
        #: arrives. Seeding from the delay rather than the interval is what
        #: preserves that.
        self.ticks_to_next_spawn = aspec.spawn_initial_delay_ticks
        self.spawned = 0

    def tick(self) -> bool:
        """Advance one tick. Returns True if the effect applies this tick."""
        applies = False
        if self.applications >= self.aspec.max_applications:
            # Spent. It still occupies the board for the rest of its life --
            # Lightning's flash lingers past its last bolt -- but it has
            # nothing left to deliver.
            self.ticks_left -= 1
            if self.ticks_left <= 0:
                self.kill()
            return False
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

    def tick_spawn(self) -> bool:
        """Whether this effect should produce a unit on this tick."""
        if not self.aspec.spawns_over_time:
            return False
        if self.ticks_to_next_spawn > 0:
            self.ticks_to_next_spawn -= 1
            return False
        self.ticks_to_next_spawn = self.aspec.spawn_interval_ticks
        self.spawned += 1
        return True

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
