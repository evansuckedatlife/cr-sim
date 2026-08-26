"""The tick loop.

The single most consequential decision in the engine is the *order* of work
within a tick. Whether targeting runs before or after movement decides whether a
unit that just came into range attacks this tick or next; whether deaths resolve
before or after projectiles decides whether a dying unit's shot still lands.
These are not implementation details -- they are the mechanics.

So the order lives in one place, as a named list, rather than being implicit in
the sequence of statements. :attr:`Battle.PHASES` can be reordered and the
interaction suite re-run to find out which orderings are load-bearing, which is
what turns calibration against real gameplay into a directed search rather than
guesswork.

Milestone M1 implements the loop, elixir, deployment, movement and win
conditions. Targeting, combat, projectiles and collision are stubs that name
what they will do; each later milestone fills one in without disturbing the
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..data.cards import Card, CardKind, CardRegistry
from ..data.leveling import LevelTable, build_tower_scales, tower_class_for
from ..data.source import LogicData
from ..replay import Command, state_hash
from .arena import Arena, load_arena
from .constants import TickClock
from .elixir import BattleTimeline, ElixirBar, build_timeline
from .entity import Entity, EntityKind, EntityState, Team, reset_entity_ids
from .fixed import distance, milli_tiles, pack_offsets, ring_offsets
from .combat import (
    AttackState,
    DamageEvent,
    PendingHit,
    advance_attack,
    apply_area_damage,
    apply_hit,
)
from .pathing import Route, crosses_river, route_to, step_towards
from .areaeffects import AreaEffect, AreaEffectSpec, build_area_effect_spec
from .projectiles import Projectile, ProjectileSpec, build_projectile_spec
from .spells import SpellPlan, plan_spell
from .targeting import acquire_target, in_attack_range, should_keep_target
from .movement import resolve_collisions
from .rng import Rng
from .spatial import SpatialIndex
from .specs import UnitSpec, build_tower_spec, build_unit_spec

__all__ = ["Battle", "BattleConfig", "Player", "BattleResult", "KING_ACTIVATION_MS"]

#: How long a provoked King Tower takes to join the fight. This build replaced
#: the old ``KING_ACTIVATE_TIME_MS`` global with an ``ActionWithDuration`` of
#: this length inside ``BUILDING.KingTower``'s action graph.
KING_ACTIVATION_MS = 3300


@dataclass(slots=True)
class Player:
    """One side's elixir, deck and towers."""

    team: Team
    deck: tuple[str, ...]
    elixir: ElixirBar
    level: int = 11
    crowns: int = 0
    #: Card names in draw order; the hand is the first four.
    cycle: list[str] = field(default_factory=list)

    @property
    def hand(self) -> tuple[str, ...]:
        return tuple(self.cycle[:4])

    @property
    def next_card(self) -> str | None:
        return self.cycle[4] if len(self.cycle) > 4 else None

    def play(self, card: str) -> bool:
        """Move ``card`` from hand to the back of the cycle."""
        if card not in self.hand:
            return False
        self.cycle.remove(card)
        self.cycle.append(card)
        return True


@dataclass(frozen=True, slots=True)
class BattleConfig:
    seed: int = 0
    ticks_per_second: int = 60
    blue_deck: tuple[str, ...] = ()
    red_deck: tuple[str, ...] = ()
    level: int = 11
    #: Crown Towers level independently of cards, on their own progression.
    tower_level: int = 11
    record_frames: bool = False
    #: Record one viewer frame every N ticks. Viewing does not need 60fps, and
    #: a full match at every tick is several megabytes of JSON.
    frame_interval: int = 3


@dataclass(frozen=True, slots=True)
class BattleResult:
    winner: Team | None
    blue_crowns: int
    red_crowns: int
    ticks: int
    reason: str


class Battle:
    """One match."""

    #: The intra-tick order. Reordering this changes mechanics, deliberately.
    PHASES: tuple[str, ...] = (
        "regenerate_elixir",
        "apply_commands",
        "fire_pending_waves",
        "advance_deploy_timers",
        "advance_lifetimes",
        "update_buffs",
        "acquire_targets",
        "resolve_attacks",
        "advance_projectiles",
        "tick_area_effects",
        "move_units",
        "resolve_collisions",
        "resolve_deaths",
        "check_tower_activation",
        "check_victory",
    )

    __slots__ = (
        "config",
        "clock",
        "arena",
        "timeline",
        "rng",
        "data",
        "levels",
        "registry",
        "entities",
        "players",
        "tick",
        "finished",
        "result",
        "_routes",
        "_specs",
        "_pending",
        "_phase_fns",
        "frames",
        "_towers",
        "_last_hands",
        "_attacks",
        "_range_extension",
        "_tower_sight_bonus",
        "damage_log",
        "_king_active",
        "_king_waking",
        "_index",
        "_max_radius",
        "_max_sight",
        "_by_id_map",
        "graveyard",
        "_projectile_specs",
        "_pre_tick_crowns",
        "_area_specs",
        "_spell_plans",
        "_pending_waves",
    )

    def __init__(
        self,
        data: LogicData,
        levels: LevelTable,
        registry: CardRegistry,
        config: BattleConfig | None = None,
        *,
        arena: Arena | None = None,
    ) -> None:
        self.config = config or BattleConfig()
        self.clock = TickClock(self.config.ticks_per_second)
        self.data = data
        self.levels = levels
        self.registry = registry
        self.arena = arena or load_arena(data)
        self.timeline: BattleTimeline = build_timeline(data, clock=self.clock)
        self.rng = Rng(self.config.seed)

        reset_entity_ids()
        self.entities: list[Entity] = []
        self._routes: dict[int, Route] = {}
        self._specs: dict[str, UnitSpec] = {}
        #: Towers indexed by team. There are only ever three a side, and every
        #: moving unit needs them each tick, so scanning the whole entity list
        #: for them turns movement into an O(entities^2) phase.
        self._towers: dict[Team, list[Entity]] = {Team.BLUE: [], Team.RED: []}
        self._attacks: dict[int, AttackState] = {}
        self.damage_log: list[DamageEvent] = []
        #: A King Tower is inert until provoked; see _phase_check_tower_activation.
        self._king_active: dict[Team, bool] = {Team.BLUE: False, Team.RED: False}
        self._king_waking: dict[Team, int] = {Team.BLUE: 0, Team.RED: 0}
        # Neighbour queries drive both targeting and collision; without an
        # index each is a full scan per unit per tick.
        self._index = SpatialIndex(self.arena.width, self.arena.height)
        self._max_radius = 0
        self._max_sight = 0
        # Id -> entity. Targeting and combat both resolve a target id every
        # tick for every unit; scanning the entity list for it is O(n^2).
        self._by_id_map: dict[int, Entity] = {}
        #: Dead entities, kept out of the live list so phases stop paying
        #: for them. A three-minute match kills hundreds.
        self.graveyard: list[Entity] = []
        self._projectile_specs: dict[str, ProjectileSpec | None] = {}
        self._area_specs: dict[str, AreaEffectSpec | None] = {}
        self._spell_plans: dict[str, SpellPlan] = {}
        #: Volleys still to fire, as (tick, card, team, x, y). Arrows is the
        #: only card that uses this, and it is why its damage looks uneven.
        self._pending_waves: list[tuple[int, str, Team, int, int]] = []
        _globals = self.data.globals_map()
        # Grace band that stops a unit flickering between two equidistant enemies.
        _ext = _globals.get('LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET', 0)
        self._range_extension = milli_tiles(_ext if isinstance(_ext, int) else 0)
        _bonus = _globals.get('EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS', 0)
        self._tower_sight_bonus = milli_tiles(_bonus if isinstance(_bonus, int) else 0)
        self._pending: list[Command] = []
        self.frames: list[dict] = []
        self._last_hands = None
        self.tick = 0
        self.finished = False
        self.result: BattleResult | None = None
        #: Both sides' crown counts as of the start of the tick currently being
        #: processed. Sudden death needs to know whether *this* tick is the one
        #: that scored, and comparing against a snapshot taken before the
        #: tick's combat runs is the only way to tell that apart from a crown
        #: that was already on the board.
        self._pre_tick_crowns: tuple[int, int] = (0, 0)

        self.players = {
            Team.BLUE: self._make_player(Team.BLUE, self.config.blue_deck),
            Team.RED: self._make_player(Team.RED, self.config.red_deck),
        }
        self._spawn_towers()
        self._phase_fns: tuple[Callable[[], None], ...] = tuple(
            getattr(self, f"_phase_{name}") for name in self.PHASES
        )

    # ------------------------------------------------------------------ setup

    def _make_player(self, team: Team, deck: Sequence[str]) -> Player:
        cycle = list(deck)
        # The opening hand is shuffled; the cycle order after that is fixed.
        self.rng.stream(f"deck:{team.name}").shuffle(cycle)
        return Player(
            team=team,
            deck=tuple(deck),
            elixir=ElixirBar(self.timeline),
            level=self.config.level,
            cycle=cycle,
        )

    def _spawn_towers(self) -> None:
        """Place both sides' structures.

        Towers use their own progression (:class:`TowerScale`), not the card
        ladder, and the King and Princess towers scale at different rates.
        """
        scales = build_tower_scales(self.data.globals_map())

        for placement in self.arena.towers:
            try:
                spec = build_tower_spec(
                    self.data,
                    placement.name,
                    scales[tower_class_for(placement.name)],
                    level=self.config.tower_level,
                    clock=self.clock,
                )
            except Exception:
                continue
            tower = Entity(
                kind=EntityKind.TOWER,
                team=placement.team,
                x=placement.x,
                y=placement.y,
                hitpoints=spec.hitpoints,
                spec=spec,
                collision_radius=spec.collision_radius,
                mass=1000,
            )
            self._register(tower)
            self._towers[placement.team].append(tower)

    def _summon_layout(self, card: Card) -> tuple[tuple[int, int], ...]:
        """Where each of a card's units lands relative to the drop point.

        A card's own ``SummonRadius`` wins where it has one. Four multi-unit
        cards ship none at all -- Skeleton Army, Minions, Archers,
        Skeleton Warriors -- and a ring is derived from how much space the units
        need instead, because fifteen skeletons cannot share a point.
        """
        total = sum(n for _name, n in card.summons())
        if total <= 1:
            return ((0, 0),) * max(1, total)

        radius = milli_tiles(card.summon_radius)
        unit_radius = 0
        for character, _count in card.summons():
            try:
                unit_radius = max(unit_radius, self._spec(character, rarity=card.rarity).collision_radius)
            except Exception:
                continue
        if radius <= 0:
            return pack_offsets(total, unit_radius or milli_tiles(500))
        # A stated radius that cannot physically hold the group still needs
        # packing, or the units simply spawn inside one another.
        import math

        if unit_radius and 2 * math.pi * radius < total * 2 * unit_radius:
            return pack_offsets(total, unit_radius)
        return ring_offsets(total, radius)

    def _spec(self, name: str, *, rarity: str) -> UnitSpec:
        key = f"{name}@{rarity}"
        spec = self._specs.get(key)
        if spec is None:
            scale = self.levels.get(rarity)
            spec = build_unit_spec(
                self.data,
                self.levels,
                name,
                level=scale.internal_level(self.config.level),
                rarity=rarity,
                clock=self.clock,
            )
            self._specs[key] = spec
        return spec

    # ------------------------------------------------------------------ input

    def queue(self, command: Command) -> None:
        self._pending.append(command)

    def play_card(self, team: Team, card_name: str, x: int, y: int) -> bool:
        """Deploy immediately, if it is legal and affordable."""
        player = self.players[team]
        card: Card | None = self.registry.get(card_name)
        if card is None or card_name not in player.hand:
            return False
        if not player.elixir.can_afford(card.mana_cost):
            return False
        if not self.arena.can_deploy(team, x, y):
            return False

        player.elixir.spend(card.mana_cost)
        player.play(card_name)
        if card.kind is CardKind.SPELL:
            self._cast(team, card, x, y)
        else:
            self._deploy(team, card, x, y)
        return True

    def _deploy(self, team: Team, card: Card, x: int, y: int) -> None:
        """Place a card's units, spread and staggered as the card specifies.

        Two details that look cosmetic but are not. ``SummonRadius`` spaces a
        swarm into a ring instead of stacking it on one point -- stacked units
        would be perfectly overlapped, so any splash would always catch the
        whole group. ``SummonDeployDelay`` staggers their arrival by 100-200ms
        each, so a swarm materialises in sequence rather than all at once.
        """
        explicit = card.summon_offsets
        group = self._summon_layout(card)
        index = 0
        for character, count in card.summons():
            try:
                spec = self._spec(character, rarity=card.rarity)
            except Exception:
                continue
            for _ in range(count):
                ox, oy = group[index] if index < len(group) else (0, 0)
                if index < len(explicit):
                    # Explicit per-unit offsets are milli-tiles, mirrored for
                    # the far side of the board.
                    ox, oy = explicit[index]
                    ox, oy = ox * 18, oy * 18 * (1 if team is Team.BLUE else -1)
                unit = Entity(
                    kind=spec.kind,
                    team=team,
                    x=x + ox,
                    y=y + oy,
                    hitpoints=spec.hitpoints,
                    spec=spec,
                    spawn_tick=self.tick,
                    deploy_ticks=spec.deploy_ticks
                    + index * self.clock.ticks(card.summon_deploy_delay),
                    collision_radius=spec.collision_radius,
                    mass=spec.mass,
                    flying=spec.flying,
                    shield=spec.shield_hitpoints,
                    lifetime_ticks=spec.lifetime_ticks,
                )
                self._register(unit)
                index += 1

    # ------------------------------------------------------------- the loop

    def step(self) -> None:
        """Advance exactly one tick, running every phase in order."""
        if self.finished:
            return
        # Rebuilt before any phase reads it, so every phase in a tick sees one
        # consistent snapshot of where things are.
        self._index.rebuild(self.entities)
        # Taken before combat runs, so check_victory can tell "a tower died
        # this tick" apart from "a tower is already dead" -- the distinction
        # sudden death is built on.
        self._pre_tick_crowns = (
            self.players[Team.BLUE].crowns,
            self.players[Team.RED].crowns,
        )
        for phase in self._phase_fns:
            phase()
        self.tick += 1
        if self.config.record_frames and self.tick % max(1, self.config.frame_interval) == 0:
            self._capture_frame()

    def _hand_delta(self) -> dict:
        """Hands, but only when they change.

        Hands are identical on the vast majority of ticks, and repeating them in
        every frame tripled the replay file for no information. The viewer
        carries the last value forward.
        """
        hands = [list(self.players[Team.BLUE].hand), list(self.players[Team.RED].hand)]
        nxt = [self.players[Team.BLUE].next_card, self.players[Team.RED].next_card]
        current = (hands, nxt)
        if getattr(self, "_last_hands", None) == current:
            return {}
        self._last_hands = current
        return {"h": hands, "n": nxt}

    def _capture_frame(self) -> None:
        """Snapshot the board for the replay viewer.

        Frames are cosmetic and never hashed, so recording them cannot change a
        battle's outcome -- only its memory use.
        """
        self.frames.append(
            {
                "t": self.tick,
                "e": [
                    [
                        e.id,
                        int(e.team),
                        int(e.kind),
                        e.x,
                        e.y,
                        e.hitpoints,
                        e.max_hitpoints,
                        1 if e.is_deploying else 0,
                        getattr(e.spec, "name", None)
                        or getattr(getattr(e, "pspec", None), "name", "?"),
                        # Real collision radius, so the viewer can draw each
                        # unit at its true footprint instead of a token dot.
                        e.collision_radius,
                    ]
                    for e in self.entities
                    if not e.dead
                ],
                "x": [self.players[Team.BLUE].elixir.exact, self.players[Team.RED].elixir.exact],
                **self._hand_delta(),
            }
        )

    def run(self, max_ticks: int | None = None) -> BattleResult:
        limit = max_ticks if max_ticks is not None else self.timeline.total_ticks
        while not self.finished and self.tick < limit:
            self.step()
        if self.result is None:
            self.result = self._decide("time")
        return self.result

    @property
    def in_overtime(self) -> bool:
        """Whether the match has crossed the regulation/overtime boundary.

        A separate read of :meth:`BattleTimeline.is_overtime` rather than a
        flag set once, so it always reflects ``self.tick`` even if a caller
        pokes the clock directly (as the M4 tests do, to reach overtime
        without simulating three minutes of ticks).
        """
        return self.timeline.is_overtime(self.tick)

    def hash(self) -> int:
        return state_hash(
            self.tick,
            self.entities,
            extra=(
                self.players[Team.BLUE].elixir.amount,
                self.players[Team.RED].elixir.amount,
            ),
        )

    # ------------------------------------------------------------------ phases

    def _phase_regenerate_elixir(self) -> None:
        for player in self.players.values():
            player.elixir.regenerate(self.tick)

    def _phase_apply_commands(self) -> None:
        if not self._pending:
            return
        due = [c for c in self._pending if c.tick <= self.tick]
        if not due:
            return
        self._pending = [c for c in self._pending if c.tick > self.tick]
        for command in due:
            self.play_card(Team(command.team), command.card, command.x, command.y)

    def _phase_fire_pending_waves(self) -> None:
        """Fire the volleys queued by a waved spell."""
        if not self._pending_waves:
            return
        due = [w for w in self._pending_waves if w[0] <= self.tick]
        if not due:
            return
        self._pending_waves = [w for w in self._pending_waves if w[0] > self.tick]
        for _tick, card_name, team, x, y in due:
            card = self.registry.get(card_name)
            if card is None:
                continue
            plan = self._plan(card)
            level = self.levels.get(card.rarity).internal_level(self.config.level)
            self._fire_at_point(team, card, plan, x, y, level)

    def _phase_advance_deploy_timers(self) -> None:
        for entity in self.entities:
            if entity.deploy_ticks_left > 0:
                entity.tick_deploy()

    def _phase_advance_lifetimes(self) -> None:
        """Expire temporary buildings.

        Spawned buildings are on a clock independent of combat: a Cannon lives
        30 seconds and a Goblin Drill 10, whether or not anything ever attacks
        them. Placing a building is therefore always a trade of permanent
        elixir for temporary board presence, which is why they can never
        just be left down.
        """
        for entity in self.entities:
            if not entity.dead and entity.lifetime_left > 0 and not entity.is_deploying:
                entity.tick_lifetime()

    def _phase_update_buffs(self) -> None:
        """M5: tick rage/slow/freeze/stun durations and expire them."""

    def _phase_acquire_targets(self) -> None:
        """Choose each unit's target, keeping the current one where possible.

        Targets are sticky. Re-choosing every tick would make units flicker
        between equidistant enemies and would erase the cost of being
        distracted, which is a real tactical currency in this game.
        """
        for entity in self.entities:
            spec = entity.spec
            if entity.dead or entity.is_deploying or spec is None:
                continue
            if entity.kind is EntityKind.TOWER and not self._can_tower_fight(entity):
                entity.target_id = 0
                continue
            if spec.hit_speed_ticks <= 0 and spec.damage <= 0:
                continue

            current = self._entity(entity.target_id)
            if should_keep_target(
                spec, entity, current, range_extension=self._range_extension
            ):
                continue

            # Only entities whose cells overlap the sight circle are even
            # considered; scanning the whole board here is what made a crowded
            # match take twenty seconds.
            reach = spec.sight_range + self._tower_sight_bonus + entity.collision_radius
            found = acquire_target(
                spec,
                entity,
                self._index.candidates(entity, reach),
                sight_bonus_for_towers=self._tower_sight_bonus,
            )
            if found is None:
                if entity.target_id and entity.id in self._attacks:
                    self._attacks[entity.id].disengage()
                entity.target_id = 0
                continue
            entity.target_id = found.id

    def _phase_resolve_attacks(self) -> None:
        """Run each engaged unit's attack cycle, then apply the tick's hits."""
        pending: list[PendingHit] = []
        for entity in self.entities:
            spec = entity.spec
            if entity.dead or entity.is_deploying or spec is None or not entity.target_id:
                continue
            if entity.hitpoints <= 0:
                continue  # fatally hit earlier this tick; it does not get to swing
            target = self._entity(entity.target_id)
            if target is None or target.dead:
                continue
            if not in_attack_range(spec, entity, target):
                # Out of reach the windup does not run at all, which is why
                # kiting a melee unit prevents its damage rather than delaying it.
                continue
            state = self._attacks.get(entity.id)
            if state is None:
                state = self._attacks[entity.id] = AttackState()
            hit = advance_attack(state, spec, entity, target)
            if hit is not None:
                pending.append(hit)

        # Apply every hit decided this tick together. Doing it inline above
        # would let entity list order decide fights: the unit iterated first
        # would land the killing blow and its victim, already at zero, would be
        # skipped before swinging back. Mirrors must trade evenly.
        for hit in pending:
            launched = self._launch(hit)
            event = None if launched else apply_hit(hit, self.tick)
            if event is not None:
                self.damage_log.append(event)
            if hit.spec.kamikaze:
                # The attack *is* the death: Ice Spirit, Balloon and Wall
                # Breakers each land one hit and are consumed by it. Without
                # this they keep swinging forever, which turns a one-shot
                # utility card into a permanent damage dealer.
                hit.attacker.kill()
            if not launched and hit.spec.area_damage_radius > 0:
                self.damage_log.extend(
                    apply_area_damage(
                        hit.spec,
                        (hit.target.x, hit.target.y),
                        hit.spec.area_damage_radius,
                        [
                            e
                            for e in self._index.near(
                                hit.target.x, hit.target.y, hit.spec.area_damage_radius
                            )
                            if e.id != hit.target.id
                        ],
                        hit.attacker,
                        self.tick,
                    )
                )

    def _plan(self, card: Card) -> SpellPlan:
        plan = self._spell_plans.get(card.name)
        if plan is None:
            plan = self._spell_plans[card.name] = plan_spell(card, self.clock)
        return plan

    def _area_spec(self, name: str, rarity: str, level: int) -> AreaEffectSpec | None:
        key = f"{name}@{rarity}@{level}"
        if key not in self._area_specs:
            self._area_specs[key] = build_area_effect_spec(
                self.data, name, self.levels.get(rarity), level=level, clock=self.clock
            )
        return self._area_specs[key]

    def _cast(self, team: Team, card: Card, x: int, y: int) -> None:
        """Put a spell's payload on the board.

        A spell is cast at a *point*, never at a unit: nothing about it is bound
        to whoever happened to be standing there. That is the whole reason a
        spell can be dodged, and the reason placing one is a prediction.
        """
        plan = self._plan(card)
        level = self.levels.get(card.rarity).internal_level(self.config.level)

        if plan.summon_character:
            self._deploy(team, card, x, y)

        if plan.projectile:
            self._fire_at_point(team, card, plan, x, y, level)
            for wave in range(1, plan.waves):
                # Later volleys are queued rather than fired now, which is what
                # lets a unit leave the area between them and take less.
                self._pending_waves.append(
                    (self.tick + wave * max(1, plan.wave_interval_ticks), card.name, team, x, y)
                )
        elif plan.area_effect:
            self._place_area(team, plan.area_effect, card.rarity, level, x, y, owner_id=0)

    def _fire_at_point(
        self, team: Team, card: Card, plan: SpellPlan, x: int, y: int, level: int
    ) -> None:
        """Launch a spell's projectile toward a ground position.

        Spell projectiles are aimed at a spot, so a stationary marker stands in
        for the target. It is deliberately not any real unit: giving the shot a
        living target would make it home, and a homing Rocket is a different
        card.
        """
        pspec = self._projectile_spec(plan.projectile, card.rarity, level)
        if pspec is None:
            return
        marker = Entity(
            kind=EntityKind.AREA_EFFECT, team=team, x=x, y=y, hitpoints=1, spawn_tick=self.tick
        )
        marker.dead = True  # never simulated; it only carries the aim point
        shot = Projectile(
            pspec=pspec, team=team, x=x, y=self._cast_origin(team),
            target=marker, owner_id=0, spawn_tick=self.tick,
        )
        shot.target_x, shot.target_y = x, y
        self.entities.append(shot)
        self._by_id_map[shot.id] = shot

    def _cast_origin(self, team: Team) -> int:
        """Where a cast spell appears to come from -- its caster's back line."""
        return 0 if team is Team.BLUE else self.arena.height

    def _place_area(
        self, team: Team, name: str, rarity: str, level: int, x: int, y: int, *, owner_id: int
    ) -> AreaEffect | None:
        aspec = self._area_spec(name, rarity, level)
        if aspec is None:
            return None
        effect = AreaEffect(
            aspec=aspec, team=team, x=x, y=y, owner_id=owner_id, spawn_tick=self.tick
        )
        self.entities.append(effect)
        self._by_id_map[effect.id] = effect
        return effect

    def _projectile_spec(self, name: str, rarity: str, level: int) -> ProjectileSpec | None:
        key = f"{name}@{rarity}@{level}"
        if key not in self._projectile_specs:
            self._projectile_specs[key] = build_projectile_spec(
                self.data, name, self.levels.get(rarity), level=level, clock=self.clock
            )
        return self._projectile_specs[key]

    def _launch(self, hit) -> bool:
        """Turn a decided hit into a shot in flight. False means melee.

        A ranged attacker does not deal damage when it swings -- it commits a
        projectile, and the damage arrives when that projectile does. In the
        gap the target can move, die, or be replaced by something else, which
        is the whole reason dodging exists.
        """
        spec = hit.spec
        if not spec.projectile:
            return False
        # The projectile must scale on the same ladder as the unit that fired it.
        pspec = self._projectile_spec(spec.projectile, spec.rarity, spec.level)
        if pspec is None or pspec.speed_per_tick <= 0:
            return False  # a zero-speed projectile is an instant hit

        shot = Projectile(
            pspec=pspec,
            team=hit.attacker.team,
            x=hit.attacker.x,
            y=hit.attacker.y,
            target=hit.target,
            owner_id=hit.attacker.id,
            spawn_tick=self.tick,
        )
        self.entities.append(shot)
        self._by_id_map[shot.id] = shot
        return True

    def _phase_advance_projectiles(self) -> None:
        """Advance every shot, and resolve the ones that land this tick.

        Impacts are collected and applied together for the same reason attacks
        are: two shots arriving on the same tick must both count, regardless of
        which happens to sit earlier in the entity list.
        """
        arrivals: list[Projectile] = []
        for entity in self.entities:
            if entity.kind is not EntityKind.PROJECTILE or entity.dead:
                continue
            if entity.advance(self._entity(entity.target_id)):
                arrivals.append(entity)

        for shot in arrivals:
            self._impact(shot)
            shot.kill()

    def _impact(self, shot: Projectile) -> None:
        """Deliver a projectile's payload where it landed."""
        pspec = shot.pspec
        if pspec.area_effect:
            # Lightning and friends: the shot is only the delivery, and what it
            # leaves behind is the actual spell.
            self._place_area(
                shot.team, pspec.area_effect, "Common", 11, shot.x, shot.y,
                owner_id=shot.owner_id,
            )
        attacker = self._entity(shot.owner_id)
        if pspec.is_splash:
            # Splash cares about where the shot *landed*, not who it was aimed
            # at -- which is why a Bomber punishes a clump even if its intended
            # target dies mid-flight.
            for victim in list(self._index.near(shot.x, shot.y, pspec.radius)):
                if victim.dead or victim.team is shot.team:
                    continue
                if victim.kind is EntityKind.PROJECTILE or not victim.is_targetable:
                    continue
                if victim.flying and not pspec.aoe_to_air:
                    continue
                if not victim.flying and not pspec.aoe_to_ground:
                    continue
                if distance(shot.x, shot.y, victim.x, victim.y) > pspec.radius + victim.collision_radius:
                    continue
                self._deal(pspec, attacker, victim)
            return

        target = self._entity(shot.target_id)
        if target is None or target.dead or target.team is shot.team:
            return  # the shot arrived at a corpse; it is simply spent
        self._deal(pspec, attacker, target)

    def _deal(self, pspec: ProjectileSpec, attacker: Entity | None, victim: Entity) -> None:
        amount = pspec.damage_to(victim.kind is EntityKind.TOWER)
        dealt = victim.apply_damage(amount)
        if dealt:
            self.damage_log.append(
                DamageEvent(
                    tick=self.tick,
                    attacker_id=attacker.id if attacker is not None else 0,
                    target_id=victim.id,
                    amount=dealt,
                    lethal=victim.hitpoints <= 0,
                )
            )

    def _phase_tick_area_effects(self) -> None:
        """Apply every live area effect to whatever is inside it right now.

        Membership is re-evaluated every application rather than captured when
        the effect was cast: a unit that walks out of a Poison cloud stops
        taking it, and one that walks in starts. Binding the victims at cast
        time would turn every lingering spell into a delayed burst.
        """
        for entity in self.entities:
            if entity.kind is not EntityKind.AREA_EFFECT or entity.dead:
                continue
            if not isinstance(entity, AreaEffect):
                continue
            if not entity.tick():
                continue
            aspec = entity.aspec
            for victim in list(self._index.near(entity.x, entity.y, aspec.radius)):
                if not entity.affects(victim):
                    continue
                if distance(entity.x, entity.y, victim.x, victim.y) > aspec.radius + victim.collision_radius:
                    continue
                if aspec.damage:
                    dealt = victim.apply_damage(
                        aspec.damage_to(victim.kind is EntityKind.TOWER)
                    )
                    if dealt:
                        self.damage_log.append(
                            DamageEvent(
                                tick=self.tick,
                                attacker_id=entity.owner_id,
                                target_id=victim.id,
                                amount=dealt,
                                lethal=victim.hitpoints <= 0,
                            )
                        )

    def _phase_move_units(self) -> None:
        for entity in self.entities:
            if entity.dead or entity.is_deploying or entity.kind is not EntityKind.TROOP:
                continue
            if entity.hitpoints <= 0:
                continue
            spec = entity.spec
            if spec is None or spec.speed_per_tick <= 0:
                continue

            state = self._attacks.get(entity.id)
            if state is not None and not state.can_move:
                continue

            target = self._entity(entity.target_id)
            if target is not None and not target.dead:
                if in_attack_range(spec, entity, target):
                    # In reach: stop and fight. Units standing still to attack is
                    # what makes a push advance at its tank's pace.
                    self._routes.pop(entity.id, None)
                    continue
                goal = (target.x, target.y)
                if entity.flying or not crosses_river(self.arena, entity.y, target.y):
                    # Same side of the water (or airborne): walk straight at it.
                    # Building a route here would mean rebuilding it every tick,
                    # since the destination moves.
                    self._routes.pop(entity.id, None)
                    self._place(
                        entity,
                        *step_towards((entity.x, entity.y), goal, spec.speed_per_tick),
                    )
                else:
                    # Across the water: this genuinely needs a bridge, and the
                    # plan survives until the crossing is done.
                    route = self._routes.get(entity.id)
                    if route is None or route.finished:
                        route = route_to(
                            self.arena, (entity.x, entity.y), goal, flying=entity.flying
                        )
                        self._routes[entity.id] = route
                    self._place(
                        entity,
                        *route.advance((entity.x, entity.y), spec.speed_per_tick),
                    )
                entity.set_state(EntityState.MOVING)
                continue

            route = self._routes.get(entity.id)
            if route is not None and route.finished:
                # Arrived. Hold position rather than re-planning every tick --
                # in M2 this is where the unit starts attacking. Only a target
                # that has died is worth re-planning for.
                target = self._by_id(entity.target_id)
                if target is not None and not target.dead:
                    entity.set_state(EntityState.IDLE)
                    continue
                route = None
                self._routes.pop(entity.id, None)
            if route is None:
                goal = self._pick_objective(entity)
                if goal is None:
                    continue
                route = route_to(
                    self.arena, (entity.x, entity.y), goal, flying=entity.flying
                )
                self._routes[entity.id] = route
            self._place(entity, *route.advance((entity.x, entity.y), spec.speed_per_tick))
            entity.set_state(EntityState.MOVING)

    def _place(self, entity: Entity, x: int, y: int) -> None:
        """Move an entity, refusing any step that would put it in terrain.

        This is the last line of defence rather than the only one: routing is
        supposed to keep ground units on bridges, and mostly does. But movement
        has several sources -- a route, a straight chase, a shove from a crowd --
        and any one of them getting it wrong puts a troop in the river. Checking
        once, here, where every move lands, makes "ground units do not stand on
        water" an invariant of the engine instead of a property each caller has
        to remember.

        A blocked diagonal falls back to whichever single axis is legal, so a
        unit slides along a bank rather than sticking to it.
        """
        if self.arena.is_walkable(x, y, flying=entity.flying):
            entity.x, entity.y = x, y
            return
        if self.arena.is_walkable(x, entity.y, flying=entity.flying):
            entity.x = x
            return
        if self.arena.is_walkable(entity.x, y, flying=entity.flying):
            entity.y = y

    def _phase_resolve_collisions(self) -> None:
        """Separate overlapping units.

        Resolved after movement rather than by blocking it: refusing to move on
        contact deadlocks a crowd, because everyone waits for space that only
        appears once somebody moves. Overlap-then-separate always converges.
        """
        if self._max_radius <= 0:
            return
        # The index was built at the top of the tick; movement has since
        # invalidated it, so refresh before asking who overlaps whom.
        self._index.rebuild(self.entities)
        resolve_collisions(self._index, self.arena, max_radius=self._max_radius)

    def _phase_resolve_deaths(self) -> None:
        """Finalise deaths and retire the bodies.

        Dead entities move to the graveyard rather than lingering in the live
        list. Every phase iterates that list, and a three-minute match kills
        hundreds of units -- left in place they would go on costing a scan each
        for the rest of the match. They stay reachable by id so a stale target
        reference resolves to something dead rather than to nothing.
        """
        died = False
        for entity in self.entities:
            if entity.state is EntityState.DYING and not entity.dead:
                entity.dead = True
                entity.state = EntityState.DEAD
                self._routes.pop(entity.id, None)
                self._attacks.pop(entity.id, None)
                died = True
                if entity.kind is EntityKind.TOWER:
                    self.players[entity.team.opponent].crowns += 1
        if died:
            alive: list[Entity] = []
            for entity in self.entities:
                if entity.dead:
                    self.graveyard.append(entity)
                else:
                    alive.append(entity)
            self.entities = alive

    def _phase_check_tower_activation(self) -> None:
        """Wake a King Tower once it is provoked.

        A King Tower sits inert at the start of a match. It joins in only when
        it takes damage directly or when one of its Princess Towers falls --
        which is why chip damage onto the King is a real commitment, and why
        losing a tower changes the whole defensive geometry of that side.

        Waking is not instant. The old ``KING_ACTIVATE_TIME_MS`` global is gone
        from this build, replaced by an action graph on ``BUILDING.KingTower``:
        an ``ActionWaitToActivate`` whose condition is
        ``king_tower_damaged() || coop_king_tower_damaged() || tower_destroyed()
        || coop_tower_destroyed()``, followed by an ``ActionWithDuration`` of
        **3300ms** during which the tower is tagged ``ACTIVATING``. That delay is
        why a tower trade can be finished before the King ever fires.
        """
        for team in (Team.BLUE, Team.RED):
            if self._king_active[team]:
                continue
            king = self._king(team)
            if king is None or king.dead:
                continue
            if self._king_waking[team] > 0:
                self._king_waking[team] -= 1
                if self._king_waking[team] == 0:
                    self._king_active[team] = True
                continue
            provoked = king.hitpoints < king.max_hitpoints or any(
                t.dead for t in self._towers[team] if t is not king
            )
            if provoked:
                self._king_waking[team] = max(1, self.clock.ticks(KING_ACTIVATION_MS))

    def _can_tower_fight(self, tower: Entity) -> bool:
        if "King" not in getattr(tower.spec, "name", ""):
            return True
        return self._king_active[tower.team]

    def _phase_check_victory(self) -> None:
        """Everything that can end a match, checked in the order the rules rank.

        Three crowns and a destroyed King are instant wins in any period --
        regulation, sudden death or the last tick of overtime -- so they are
        checked first and unconditionally, exactly as before M4. Only once
        neither has fired does the period matter: sudden death turns *any*
        tower kill into an instant win once overtime has started, regulation
        ending level sends the match into overtime rather than ending it, and
        overtime running out with nothing decided falls to the tiebreaker.
        """
        for team in (Team.BLUE, Team.RED):
            if self.players[team].crowns >= 3:
                self.result = self._decide("three crowns")
                self.finished = True
                return
            king = self._king(team)
            if king is not None and king.dead:
                self.result = self._decide("king tower destroyed")
                self.finished = True
                return

        if self.timeline.is_overtime(self.tick):
            blue_before, red_before = self._pre_tick_crowns
            blue_scored = self.players[Team.BLUE].crowns > blue_before
            red_scored = self.players[Team.RED].crowns > red_before
            if blue_scored != red_scored:
                # Exactly one side landed a tower this tick -- sudden death:
                # the very first tower destroyed in overtime wins outright,
                # Princess Tower or King alike (a King death was already
                # caught above, with its own reason).
                self.result = self._decide("sudden death")
                self.finished = True
                return
            if self.tick == self.timeline.total_ticks - 1:
                # Overtime's last tick resolved with nobody having scored --
                # crowns are guaranteed level here, since any period that put
                # them ahead would have ended the match already.
                self.result = self._decide_tiebreaker()
                self.finished = True
                return
        elif self.tick == self.timeline.regulation_ticks - 1:
            # The last tick of regulation. Level, the match plays on into
            # overtime -- nothing to do here but let the clock cross the
            # boundary. Ahead, it ends right now.
            blue = self.players[Team.BLUE].crowns
            red = self.players[Team.RED].crowns
            if blue != red:
                self.result = self._decide("regulation")
                self.finished = True
                return

    # ------------------------------------------------------------- helpers

    def _entity(self, entity_id: int) -> Entity | None:
        """Look up any entity by id, alive or dead."""
        return self._by_id_map.get(entity_id) if entity_id else None

    def _register(self, entity: Entity) -> None:
        self.entities.append(entity)
        self._by_id_map[entity.id] = entity
        spec = entity.spec
        if spec is not None:
            if spec.collision_radius > self._max_radius:
                self._max_radius = spec.collision_radius
            if spec.sight_range > self._max_sight:
                self._max_sight = spec.sight_range

    def _by_id(self, entity_id: int) -> Entity | None:
        if not entity_id:
            return None
        for tower in self._towers[Team.BLUE] + self._towers[Team.RED]:
            if tower.id == entity_id:
                return tower
        return None

    def _pick_objective(self, entity: Entity) -> tuple[int, int] | None:
        """Choose and remember where a unit is walking.

        Ground troops head for the enemy Princess Tower on their side of the
        board, which is what makes a unit's deployment x-position decide its
        lane. The King Tower only becomes the objective once the Princess Tower
        covering that side is gone.
        """
        candidates = [t for t in self._towers[entity.team.opponent] if not t.dead]
        if not candidates:
            return None
        princesses = [e for e in candidates if "King" not in getattr(e.spec, "name", "")]
        pool = princesses or candidates
        # Nearest by x keeps a unit in the lane it was placed in; the id is a
        # stable tiebreak so two equidistant units never disagree.
        best = min(pool, key=lambda e: (abs(e.x - entity.x), e.id))
        entity.target_id = best.id
        return best.x, best.y

    def _king(self, team: Team) -> Entity | None:
        """The team's King Tower, alive or dead.

        Read from the tower index rather than the live entity list: a destroyed
        King is retired to the graveyard, and the victory check needs to be able
        to see that it died.
        """
        for entity in self._towers[team]:
            if "King" in getattr(entity.spec, "name", ""):
                return entity
        return None

    def _decide(self, reason: str) -> BattleResult:
        blue = self.players[Team.BLUE].crowns
        red = self.players[Team.RED].crowns
        winner: Team | None
        if blue > red:
            winner = Team.BLUE
        elif red > blue:
            winner = Team.RED
        else:
            winner = None
        return BattleResult(
            winner=winner, blue_crowns=blue, red_crowns=red, ticks=self.tick, reason=reason
        )

    def _worst_princess(self, team: Team) -> tuple[int, int]:
        """``team``'s most-damaged Princess Tower, as ``(hitpoints, max_hitpoints)``.

        A destroyed tower is a perfectly valid answer here -- zero hitpoints
        is the lowest percentage there is, so a Princess Tower already lost
        earlier in the match (the two sides can each be down one and still be
        level on crowns) outranks any tower still standing. The King never
        enters into this: the tiebreaker is expressly about Princess Towers.
        """
        worst_hp, worst_max = 0, 1
        seen = False
        for tower in self._towers[team]:
            if "King" in getattr(tower.spec, "name", ""):
                continue
            if not seen or tower.hitpoints * worst_max < worst_hp * tower.max_hitpoints:
                worst_hp, worst_max = tower.hitpoints, tower.max_hitpoints
                seen = True
        return worst_hp, worst_max

    def _decide_tiebreaker(self) -> BattleResult:
        """Resolve a match that finishes overtime still level on crowns.

        Sudden death gives an instant winner to whichever side lands the
        first tower kill; if overtime runs out without one, the match falls
        back to comparing each side's worst (most-damaged) Princess Tower as
        a *fraction of its own maximum* -- the side that damaged the
        opponent's tower more, in percentage terms, wins. Percentage rather
        than raw hitpoints, because otherwise a tower with a bigger health
        pool would look "less damaged" than an equally-battered smaller one.
        The comparison is done by cross-multiplying two fractions rather than
        dividing, so it stays exact in integer arithmetic.
        """
        blue_hp, blue_max = self._worst_princess(Team.BLUE)
        red_hp, red_max = self._worst_princess(Team.RED)
        # blue_hp/blue_max vs red_hp/red_max, without ever dividing.
        blue_share = blue_hp * red_max
        red_share = red_hp * blue_max
        if blue_share == red_share:
            # Both sides took exactly the same proportional damage. Not a
            # tie broken either way -- a genuine draw.
            return self._decide("draw")
        winner = Team.BLUE if blue_share > red_share else Team.RED
        return BattleResult(
            winner=winner,
            blue_crowns=self.players[Team.BLUE].crowns,
            red_crowns=self.players[Team.RED].crowns,
            ticks=self.tick,
            reason="tiebreaker",
        )

    # ------------------------------------------------------------------ views

    def living(self, team: Team | None = None) -> Iterable[Entity]:
        for entity in self.entities:
            if entity.dead:
                continue
            if team is None or entity.team is team:
                yield entity
