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

from ..data.cards import Card, CardRegistry
from ..data.leveling import LevelTable
from ..data.source import LogicData
from ..replay import Command, state_hash
from .arena import Arena, load_arena
from .constants import TickClock
from .elixir import BattleTimeline, ElixirBar, build_timeline
from .entity import Entity, EntityKind, EntityState, Team, reset_entity_ids
from .pathing import Route, route_to
from .rng import Rng
from .specs import UnitSpec, build_unit_spec

__all__ = ["Battle", "BattleConfig", "Player", "BattleResult"]


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
    record_frames: bool = False


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
        "advance_deploy_timers",
        "update_buffs",
        "acquire_targets",
        "resolve_attacks",
        "advance_projectiles",
        "tick_area_effects",
        "move_units",
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
        self._pending: list[Command] = []
        self.tick = 0
        self.finished = False
        self.result: BattleResult | None = None

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

        NOTE (open question ``tower-hp-scaling``): towers are currently scaled on
        the card ladder, which is probably wrong. ``globals.csv`` carries a
        separate tower progression the cards do not use --
        ``HITPOINT_INCREASE_PERCENT_PER_TOWER_LEVEL=8``,
        ``..._PER_KING_LEVEL=7``, ``..._AFTER_TOURNAMENTCAP=10`` and
        ``TOWER_SCALING_START_EXP_LEVEL=9``. This must be resolved in M2, when
        towers begin fighting and their hitpoints start deciding outcomes.
        """
        for placement in self.arena.towers:
            try:
                spec = self._spec(placement.name, rarity="Common")
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
            self.entities.append(tower)

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
        self._deploy(team, card, x, y)
        return True

    def _deploy(self, team: Team, card: Card, x: int, y: int) -> None:
        offsets = card.summon_offsets
        index = 0
        for character, count in card.summons():
            try:
                spec = self._spec(character, rarity=card.rarity)
            except Exception:
                continue
            for _ in range(count):
                ox, oy = (0, 0)
                if index < len(offsets):
                    # Offsets are milli-tiles, mirrored for the far side.
                    ox, oy = offsets[index]
                    ox, oy = ox * 18, oy * 18 * (1 if team is Team.BLUE else -1)
                unit = Entity(
                    kind=spec.kind,
                    team=team,
                    x=x + ox,
                    y=y + oy,
                    hitpoints=spec.hitpoints,
                    spec=spec,
                    spawn_tick=self.tick,
                    deploy_ticks=spec.deploy_ticks,
                    collision_radius=spec.collision_radius,
                    mass=spec.mass,
                    flying=spec.flying,
                    shield=spec.shield_hitpoints,
                )
                self.entities.append(unit)
                index += 1

    # ------------------------------------------------------------- the loop

    def step(self) -> None:
        """Advance exactly one tick, running every phase in order."""
        if self.finished:
            return
        for phase in self._phase_fns:
            phase()
        self.tick += 1

    def run(self, max_ticks: int | None = None) -> BattleResult:
        limit = max_ticks if max_ticks is not None else self.timeline.total_ticks
        while not self.finished and self.tick < limit:
            self.step()
        if self.result is None:
            self.result = self._decide("time")
        return self.result

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

    def _phase_advance_deploy_timers(self) -> None:
        for entity in self.entities:
            if entity.deploy_ticks_left > 0:
                entity.tick_deploy()

    def _phase_update_buffs(self) -> None:
        """M5: tick rage/slow/freeze/stun durations and expire them."""

    def _phase_acquire_targets(self) -> None:
        """M2: sight checks, target filters, priority and retarget rules."""

    def _phase_resolve_attacks(self) -> None:
        """M2: load/first-hit/hit-speed state machine and damage application."""

    def _phase_advance_projectiles(self) -> None:
        """M2: fly projectiles and resolve impacts."""

    def _phase_tick_area_effects(self) -> None:
        """M5: area-effect damage ticks and buff application."""

    def _phase_move_units(self) -> None:
        for entity in self.entities:
            if entity.dead or entity.is_deploying or entity.kind is not EntityKind.TROOP:
                continue
            spec = entity.spec
            if spec is None or spec.speed_per_tick <= 0:
                continue
            route = self._routes.get(entity.id)
            if route is None or route.finished:
                goal = self._objective(entity)
                if goal is None:
                    continue
                route = route_to(
                    self.arena, (entity.x, entity.y), goal, flying=entity.flying
                )
                self._routes[entity.id] = route
            entity.x, entity.y = route.advance((entity.x, entity.y), spec.speed_per_tick)
            entity.set_state(EntityState.MOVING)

    def _phase_resolve_deaths(self) -> None:
        for entity in self.entities:
            if entity.state is EntityState.DYING and not entity.dead:
                entity.dead = True
                entity.state = EntityState.DEAD
                self._routes.pop(entity.id, None)
                if entity.kind is EntityKind.TOWER:
                    self.players[entity.team.opponent].crowns += 1

    def _phase_check_tower_activation(self) -> None:
        """M2: wake the King Tower when a Princess falls or it takes damage."""

    def _phase_check_victory(self) -> None:
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

    # ------------------------------------------------------------- helpers

    def _objective(self, entity: Entity) -> tuple[int, int] | None:
        """Where a unit walks when it has no target yet.

        Ground troops head for the enemy Princess Tower on their side of the
        board, which is what makes a unit's deployment x-position decide its
        lane. The King Tower is only the objective once both Princess Towers on
        that side are gone.
        """
        enemy = entity.team.opponent
        candidates = [
            e
            for e in self.entities
            if e.team is enemy and e.kind is EntityKind.TOWER and not e.dead
        ]
        if not candidates:
            return None
        princesses = [e for e in candidates if "King" not in getattr(e.spec, "name", "")]
        pool = princesses or candidates
        # Nearest by x keeps a unit in the lane it was placed in.
        best = min(pool, key=lambda e: (abs(e.x - entity.x), e.id))
        return best.x, best.y

    def _king(self, team: Team) -> Entity | None:
        for entity in self.entities:
            if (
                entity.team is team
                and entity.kind is EntityKind.TOWER
                and "King" in getattr(entity.spec, "name", "")
            ):
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

    # ------------------------------------------------------------------ views

    def living(self, team: Team | None = None) -> Iterable[Entity]:
        for entity in self.entities:
            if entity.dead:
                continue
            if team is None or entity.team is team:
                yield entity
