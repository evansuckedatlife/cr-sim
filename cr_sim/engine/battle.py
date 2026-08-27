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

from copy import deepcopy as _deepcopy

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

from ..data.cards import Card, CardKind, CardRegistry
from ..data.leveling import LevelTable, build_tower_scales, tower_class_for
from ..data.source import LogicData
from ..replay import Command, state_hash
from .arena import Arena, TowerPlacement, load_arena
from .constants import TickClock
from .elixir import BattleTimeline, ElixirBar, build_timeline
from .entity import Entity, EntityKind, EntityState, Team, reset_entity_ids
from .fixed import (
    SUBTILES_PER_TILE,
    distance,
    milli_tiles,
    pack_offsets,
    push_away,
    ring_offsets,
)
from .combat import (
    AttackState,
    DamageEvent,
    PendingHit,
    advance_attack,
    apply_area_damage,
    apply_hit,
)
from .pathgrid import PathGrid, load_path_costs
from .pathing import (
    Route,
    crosses_river,
    line_blocked,
    route_to,
    step_towards,
)
from .areaeffects import AreaEffect, AreaEffectSpec, build_area_effect_spec
from .actions import ActionContext, ActionInterpreter
from .buffs import BuffSpec, BuffState, apply_delta, build_buff_spec
from .projectiles import (
    Projectile,
    ProjectileSpec,
    RollingProjectile,
    build_projectile_spec,
)
from .spells import SpellPlan, plan_spell
from .targeting import UNTARGETABLE_KINDS, acquire_target, in_attack_range, should_keep_target
from .movement import resolve_collisions
from .rng import Rng
from .spatial import SpatialIndex
from .specs import UnitSpec, build_tower_spec, build_unit_spec

__all__ = [
    "Battle", "BattleConfig", "Player", "BattleResult", "KING_ACTIVATION_MS",
    "CLONE_HITPOINTS",
    "MIRROR_EXTRA_ELIXIR",
]

#: What Mirror adds to the cost of the card it copies. Not in the build: the
#: card's own ManaCost is 1, which is what it would cost if it deployed
#: nothing, and the globals carry the level offset but not the price. Recorded
#: as an anchor rather than presented as read.
MIRROR_EXTRA_ELIXIR = 1

#: A clone's hitpoints. Every other number the Clone spell needs is in the
#: build -- the offset, the shield rule, whether a clone may be cloned -- but
#: this one is in none of them, so it is the card's documented behaviour rather
#: than a value read from the files. See `clone-hitpoints` in
#: reference/anchors.json.
CLONE_HITPOINTS = 1


def _int_global(globals_map: dict, key: str, default: int) -> int:
    value = globals_map.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default

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
    #: The last card actually deployed, for Mirror to replay. Mirror itself is
    #: never recorded: mirroring a Mirror has nothing to copy.
    last_played: str | None = None
    #: Card names this deck slotted as evolutions. Empty by default: having an
    #: evolution available is a deck-building choice, not a property of owning
    #: the card.
    evolutions: tuple[str, ...] = ()
    #: Plays of each evolution card since it last came out evolved. A slot
    #: starts charged, which is how a match begins: the first play of an
    #: evolution card is the evolved one.
    evolution_charge: dict[str, int] = field(default_factory=dict)

    def evolution_ready(self, card: "Card") -> bool:
        """Whether this card's next play is its evolved form.

        Only for cards the deck actually slotted as evolutions. Most of the
        roster *has* an evolution -- 42 of 122 cards -- but a deck carries at
        most a couple, and treating every card as evolved because one exists
        would quietly change the stats of half the game.

        One cycle means every second play is evolved, two means every third.
        Counted in plays of the card rather than deck rotations, which is the
        same thing: a card returns to hand once per rotation.
        """
        if card.name not in self.evolutions:
            return False
        if not card.evolution or card.evolution_cycles <= 0:
            return False
        return self.evolution_charge.get(card.name, card.evolution_cycles) >= card.evolution_cycles

    def spend_evolution(self, card: "Card") -> None:
        self.evolution_charge[card.name] = 0

    def cycle_evolution(self, card: "Card") -> None:
        if card.name in self.evolutions and card.evolution and card.evolution_cycles > 0:
            charge = self.evolution_charge.get(card.name, card.evolution_cycles)
            self.evolution_charge[card.name] = charge + 1

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
    #: Cards each side slotted as evolutions. Empty by default: most of the
    #: roster has an evolution available, but a deck carries at most a couple,
    #: and evolving every card that could would change the stats of half the
    #: game without anyone asking for it.
    blue_evolutions: tuple[str, ...] = ()
    red_evolutions: tuple[str, ...] = ()
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
        "run_actions",
        "run_spawners",
        "update_buffs",
        "update_conditional_buffs",
        "acquire_targets",
        "resolve_attacks",
        "advance_projectiles",
        "tick_area_effects",
        "pull_units",
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
        "_last_attack",
        "_spawn_timers",
        "_spawn_children",
        "_charge",
        "path_grid",
        "_occupancy_signature",
        "_hit_counts",
        "actions",
        "_buff_specs",
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
        self.actions = ActionInterpreter(
            self.data,
            self.clock,
            self._spawn_units,
            self._clone_entity,
            self._place_area_from_action,
            self._count_living,
            self._apply_buff_from_action,
        )

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
        self._buff_specs: dict[str, BuffSpec | None] = {}
        #: Volleys still to fire, as (tick, card, team, x, y). Arrows is the
        #: only card that uses this, and it is why its damage looks uneven.
        self._pending_waves: list[tuple[int, str, Team, int, int]] = []
        #: Tick of each entity's most recent swing, for the buffs that
        #: depend on how long it has been since one.
        self._last_attack: dict[int, int] = {}
        #: Ticks until each spawner's next wave.
        self._spawn_timers: dict[int, int] = {}
        #: Children of the spawners that cap how many may live at once.
        self._spawn_children: dict[int, list[int]] = {}
        #: Distance each charging unit has covered unobstructed, in
        #: subtiles. Prince, Dark Prince and Battle Ram all live on this.
        self._charge: dict[int, int] = {}
        #: Weighted movement costs, so buildings bend a push instead of
        #: being walked through. Occupancy is refreshed as they change.
        self.path_grid = PathGrid(
            self.arena, load_path_costs(self.data.globals_map())
        )
        self._occupancy_signature: tuple = ()
        #: Hits each unit has landed, for the after-hits buff ladders.
        self._hit_counts: dict[int, int] = {}
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
        # Mirror carries OmitFromStartingHand, and the reason is plain: with
        # nothing yet played it has nothing to copy, so dealing it into the
        # opening hand would be dealing a dead card.
        held = [n for n in cycle[:4] if (c := self.registry.get(n)) and c.omit_from_starting_hand]
        for name in held:
            cycle.remove(name)
            cycle.append(name)
        slotted = (
            self.config.blue_evolutions if team is Team.BLUE else self.config.red_evolutions
        )
        return Player(
            team=team,
            deck=tuple(deck),
            elixir=ElixirBar(self.timeline),
            level=self.config.level,
            cycle=cycle,
            evolutions=tuple(name for name in slotted if name in deck),
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

    def _spec(self, name: str, *, rarity: str, level_offset: int = 0) -> UnitSpec:
        key = f"{name}@{rarity}@{level_offset}"
        spec = self._specs.get(key)
        if spec is None:
            scale = self.levels.get(rarity)
            spec = build_unit_spec(
                self.data,
                self.levels,
                name,
                level=scale.internal_level(self.config.level + level_offset),
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

        # Mirror is a reference, not a card. It costs whatever it is copying
        # plus one and deploys that at a level higher, so both the price and
        # the placement rules have to come from the copied card rather than
        # from Mirror's own row -- Mirror says one elixir and "anywhere", and
        # neither is true of what it actually puts down.
        mirrored: Card | None = None
        level_offset = 0
        cost = card.mana_cost

        # A variant card is whichever form the elixir on hand pays for. Merge
        # Maiden mounted at six, on foot at three -- so the cost is not on the
        # card, it is the trigger of the form that was actually afforded.
        if card.variants:
            chosen = next(
                (
                    (price, name)
                    for price, name in card.variants
                    if player.elixir.can_afford(price)
                ),
                None,
            )
            if chosen is None:
                return False
            variant = self.registry.get(chosen[1])
            if variant is None:
                return False
            cost, card = chosen[0], variant
        if card.is_mirror:
            mirrored = self.registry.get(player.last_played or "")
            if mirrored is None or mirrored.is_mirror:
                return False
            cost = mirrored.mana_cost + MIRROR_EXTRA_ELIXIR
            level_offset = self._mirror_level_offset()
            card = mirrored

        if not player.elixir.can_afford(cost):
            return False
        # Honour the card's own placement rules. Without this a Fireball
        # cannot be cast on the enemy half at all -- the only place anyone
        # would ever cast one -- so every spell lands in its owner's own
        # territory and hits nothing.
        if not self.arena.can_deploy(
            team,
            x,
            y,
            anywhere=card.can_deploy_on_enemy_side,
            on_water=card.can_place_on_water,
            fallen_enemy_towers=self.fallen_enemy_towers(team),
        ):
            return False

        player.elixir.spend(cost)
        player.play(card_name)
        if mirrored is None:
            # Mirror never becomes the thing it would copy next.
            player.last_played = card.name

        # An evolution slot deploys a different card entirely -- Evo Barbarians
        # summons Barbarian_EV1, five of them instead of four -- while costing
        # the same and occupying the same deck slot. Swapped here rather than
        # in the deck so the cycle keeps running on the base card, which is
        # what recharges the evolution.
        played = card
        if player.evolution_ready(card):
            evolved = self.registry.get(card.evolution) if card.evolution else None
            if evolved is not None:
                played = evolved
                player.spend_evolution(card)
        else:
            player.cycle_evolution(card)

        if played.kind is CardKind.SPELL:
            self._cast(team, played, x, y)
        else:
            self._deploy(team, played, x, y, level_offset)
        return True

    def _mirror_level_offset(self) -> int:
        value = self.data.globals_map().get("MIRROR_LEVEL_OFFSET", 1)
        return value if isinstance(value, int) and not isinstance(value, bool) else 1

    def _deploy(self, team: Team, card: Card, x: int, y: int, level_offset: int = 0) -> None:
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
                spec = self._spec(character, rarity=card.rarity, level_offset=level_offset)
            except Exception:
                continue
            for _ in range(count):
                ox, oy = group[index] if index < len(group) else (0, 0)
                if index < len(explicit):
                    # Explicit per-unit offsets are milli-tiles, mirrored for
                    # the far side of the board.
                    ox, oy = explicit[index]
                    ox, oy = ox * 18, oy * 18 * (1 if team is Team.BLUE else -1)
                px, py = self._settle(x + ox, y + oy, flying=spec.flying)
                unit = Entity(
                    kind=spec.kind,
                    team=team,
                    x=px,
                    y=py,
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
                self._begin_actions(unit)
                index += 1

    # ------------------------------------------------------------ branching

    #: Everything a clone may share rather than copy. All of it is either
    #: immutable or a cache keyed by name whose entries are themselves
    #: immutable, so a branch reading through it cannot disturb the battle it
    #: came from -- and a warmed cache is worth inheriting rather than
    #: rebuilding. Copying these instead is what made a naive deepcopy cost
    #: five times more than simulating an entire match: they hold the whole
    #: card database.
    _SHARED = (
        "config", "clock", "data", "levels", "registry", "arena", "timeline",
        "_specs", "_projectile_specs", "_area_specs", "_spell_plans",
        "_buff_specs",
    )

    #: Append-only records: written during a tick, never read back by one.
    _HISTORIES = ("graveyard", "damage_log", "frames")

    def clone(self) -> "Battle":
        """An independent battle continuing from this position.

        For asking what happens next without committing to it: play the copy
        forward, read the outcome, throw it away. The original is untouched,
        and because the engine is deterministic the answer is exact rather
        than sampled.

        Implemented by seeding a deepcopy memo with the shared objects instead
        of enumerating the mutable ones. The enumeration is the version that
        rots -- a slot added later would silently stay shared between a battle
        and its branches, and the resulting bug would look like nondeterminism
        rather than like a missing copy.
        """
        memo: dict[int, object] = {}
        for name in self._SHARED:
            value = getattr(self, name)
            memo[id(value)] = value
        # The interpreter's own caches, on the same grounds: one is parsed
        # ACTION rows keyed by name, the other a tally of unimplemented class
        # types. Neither is state the simulation reads back.
        memo[id(self.actions._cache)] = self.actions._cache
        memo[id(self.actions.unsupported)] = self.actions.unsupported

        # The append-only histories are set aside rather than copied. A branch
        # adds to them and reads its own additions, but never rereads or
        # mutates what was already there, so it can share the entries and take
        # only a fresh container. On a mid-match position these three were
        # more than half the cost of a copy.
        stash = {name: getattr(self, name) for name in self._HISTORIES}
        for name in stash:
            setattr(self, name, [])
        # The corpses go in the memo as themselves, so the branch's id map
        # points at the same objects its graveyard list already does.
        #
        # Setting `graveyard` aside is not enough on its own: `_by_id_map`
        # keeps every entity ever registered reachable by id, dead included
        # (that is what makes a stale target reference resolve to a corpse
        # rather than to nothing), so deepcopy was reaching the whole
        # graveyard through the map and rebuilding every one of them. By the
        # end of a match that is a couple of hundred entities copied per
        # clone, none of which any phase ever touches -- and it left the
        # branch with two different objects for one dead unit, the shared one
        # in `graveyard` and a private one in the map. Sharing rests on
        # exactly the invariant the histories already rest on: nothing rereads
        # or mutates what is already in them. Dead entities are not in
        # `entities`, so no phase iterates them, and every path that resolves
        # one by id tests `.dead` before doing anything with it.
        for corpse in stash["graveyard"]:
            memo[id(corpse)] = corpse
        try:
            clone = _deepcopy(self, memo)
        finally:
            for name, value in stash.items():
                setattr(self, name, value)

        clone.graveyard = list(stash["graveyard"])
        clone.damage_log = list(stash["damage_log"])
        # Frames are a viewer artefact, and a branch that is thrown away has
        # no viewer. Kept off so a lookahead does not accumulate megabytes.
        clone.frames = []
        return clone

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

    def _phase_run_actions(self) -> None:
        """Advance the ACTION graph.

        Actions carry real delays -- Graveyard's twelve skeletons are authored
        at 2200ms through 8200ms -- so the interpreter holds a queue and this
        drains whatever is due. Placed before the spawners so an action that
        creates a hut has it producing on the same tick a stat-driven one would.
        """
        self.actions.drain(self.tick)

    def _phase_run_spawners(self) -> None:
        """Buildings and units that produce troops on a timer.

        Goblin Hut, Barbarian Hut, Tombstone, Furnace and Witch are all one
        mechanism. Three fields drive it and the data settles what each means:

        ``SpawnPauseTime``
            The gap between waves. Witch reads 7000ms and produces four
            Skeletons, which is exactly the seven-second cycle the card is
            known for -- that is what pins this field down rather than
            ``SpawnInterval``.
        ``SpawnNumber``
            Units per wave.
        ``SpawnInterval``
            The stagger *within* a wave, 500ms where present. Witch carries no
            ``SpawnInterval`` at all, which is why her Skeletons arrive
            together while a hut's trickle out.

        ``SpawnStartTime`` delays only the first wave; where it is absent the
        first wave waits a full cycle like every one after it.
        """
        for entity in self.entities:
            spec = entity.spec
            if spec is None or entity.dead or not spec.spawn_character:
                continue
            if entity.is_deploying:
                # A hut that has not finished landing is not producing yet.
                continue
            due = self._spawn_timers.get(entity.id)
            if due is None:
                due = spec.spawn_start_ticks or spec.spawn_pause_ticks
            if due > 0:
                self._spawn_timers[entity.id] = due - 1
                continue

            self._spawn_timers[entity.id] = max(1, spec.spawn_pause_ticks)
            if spec.spawn_limit:
                living = [
                    cid for cid in self._spawn_children.get(entity.id, ())
                    if (child := self._entity(cid)) is not None and not child.dead
                ]
                self._spawn_children[entity.id] = living
                room = spec.spawn_limit - len(living)
                if room <= 0:
                    continue
            else:
                room = spec.spawn_count

            born = self._spawn_units(
                team=entity.team,
                character=spec.spawn_character,
                count=min(spec.spawn_count, room),
                x=entity.x,
                y=entity.y,
                stagger_ticks=spec.spawn_interval_ticks,
                rarity=spec.rarity,
            )
            if spec.spawn_limit and born:
                self._spawn_children.setdefault(entity.id, []).extend(u.id for u in born)

    def _phase_update_buffs(self) -> None:
        """Expire buff durations and deliver damage-over-time.

        Poison and Tornado carry no damage of their own -- all of it lives in
        the buff they leave behind, ticking on the buff's own ``HitFrequency``
        rather than the cloud's scan rate. Ticking here rather than inside the
        area effect is what lets the damage keep landing on a unit that has
        already walked out of the cloud but is still poisoned.
        """
        for entity in self.entities:
            if entity.buffs is None or entity.dead:
                continue
            owed = entity.buffs.tick(
                entity.kind is EntityKind.TOWER, entity.kind is EntityKind.BUILDING
            )
            if owed.damage:
                dealt = entity.apply_damage(owed.damage)
                if dealt:
                    self.damage_log.append(
                        DamageEvent(
                            tick=self.tick,
                            attacker_id=0,
                            target_id=entity.id,
                            amount=dealt,
                            lethal=entity.hitpoints <= 0,
                        )
                    )
            if owed.heal and not entity.dead:
                # Healing cannot resurrect. A unit whose hitpoints already
                # reached zero this tick is dead in the same sense as one
                # killed by a sword, and the death phase will retire it.
                ceiling = entity.max_hitpoints
                if owed.over_heal_percent:
                    ceiling += ceiling * owed.over_heal_percent // 100
                entity.hitpoints = min(entity.hitpoints + owed.heal, ceiling)
            if not entity.buffs:
                entity.buffs = None

    def _phase_update_conditional_buffs(self) -> None:
        """Grant and revoke the buffs a unit holds because of what it is doing.

        ``BuffWhenNotAttacking`` is not a timed application like a spell's --
        it is a *state*. Royal Ghost is invisible for as long as it has not
        swung recently and solid the instant it does, which is the whole card:
        it cannot be blocked or targeted on the way in, and it is exposed while
        it works. The Knight evolution's damage reduction and Suspicious Bush
        work the same way.

        Re-asserted every tick rather than applied once with a duration,
        because the condition it depends on can change under it at any moment.
        """
        for entity in self.entities:
            spec = entity.spec
            if spec is None or entity.dead or not spec.buff_when_not_attacking:
                continue
            # Never having attacked counts as idle since deploy, so a Royal
            # Ghost fades in on its own two seconds after it lands.
            since = self.tick - self._last_attack.get(entity.id, entity.spawn_tick)
            # At least one tick, so a unit with a zero threshold (Suspicious
            # Bush) still drops the buff on the tick it attacks rather than
            # satisfying "has not attacked for 0ms" while mid-swing.
            threshold = max(1, spec.buff_when_not_attacking_ticks)
            if since >= threshold:
                bspec = self._buff_spec(spec.buff_when_not_attacking, spec.rarity, spec.level)
                if bspec is not None:
                    self._apply_buff(entity, bspec, 2, source=entity.id)
            elif entity.buffs is not None:
                entity.buffs.remove(spec.buff_when_not_attacking)

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
            if entity.buffs is not None and entity.buffs.is_frozen():
                # Frozen means frozen: no windup progress, no swing. This is
                # why Freeze buys time rather than merely reducing damage.
                # A stunned charger is stopped, and stopping loses the run-up.
                if entity.spec is not None and entity.spec.charge_range:
                    self._charge[entity.id] = 0
                stunned = self._attacks.get(entity.id)
                if stunned is not None and spec.load_first_hit:
                    stunned.reset_load(spec)
                if stunned is not None:
                    # And a stun sends a ramping attacker back to its first
                    # stage. An Inferno Tower does not forget its target, it
                    # starts the burn again -- which is the entire reason a
                    # 500ms Electro Wizard zap answers the card.
                    stunned.break_lock()
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
            self._last_attack[hit.attacker.id] = self.tick
            if hit.spec.charge_range:
                if self._is_charged(hit.attacker, hit.spec) and hit.spec.damage_special:
                    # The charge hit. Prince's DamageSpecial is exactly double
                    # his ordinary damage, and spending it here is what makes
                    # blocking a Prince before he connects worth a card.
                    hit.damage = hit.spec.damage_special
                # Connecting spends the charge whether or not it was ready.
                self._charge[hit.attacker.id] = 0
            if hit.spec.buff_on_damage:
                # Electro Wizard's stun. Applied on the decision to hit rather
                # than on impact, so a shot already in the air still stuns --
                # and so the stun lands even against a target that dies to the
                # same swing, which is what makes him a reliable Inferno reset.
                bspec = self._buff_spec(hit.spec.buff_on_damage, hit.spec.rarity, hit.spec.level)
                if bspec is not None:
                    self._apply_buff(
                        hit.target, bspec, hit.spec.buff_on_damage_ticks, source=hit.attacker.id
                    )
            self._after_hits(hit)
            self._reflect(hit)
            if hit.attacker.buffs is not None:
                # Swinging breaks invisibility. Royal Ghost is untargetable
                # between attacks and exposed from the moment it commits to
                # one -- without this it is untargetable for its whole life.
                hit.attacker.buffs.on_attack()
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

    def _after_hits(self, hit) -> None:
        """Grant a unit the buff it has earned by landing hits.

        Prince's ladder is 2 / 4 / 6 hits for 6000 / 4000 / 2000ms of
        escalating rage: the longer he is left swinging the worse he gets. The
        count is per unit and never resets, so it is a running total of what
        that unit has done rather than a per-target streak.
        """
        spec = hit.spec
        if not spec.buff_after_hits:
            return
        landed = self._hit_counts.get(hit.attacker.id, 0) + 1
        self._hit_counts[hit.attacker.id] = landed
        for index, threshold in enumerate(spec.buff_after_hits_count):
            if landed != threshold or index >= len(spec.buff_after_hits):
                continue
            bspec = self._buff_spec(spec.buff_after_hits[index], spec.rarity, spec.level)
            if bspec is None:
                continue
            duration = (
                spec.buff_after_hits_ticks[index]
                if index < len(spec.buff_after_hits_ticks)
                else 0
            )
            self._apply_buff(hit.attacker, bspec, duration, source=hit.attacker.id)

    def _reflect(self, hit) -> None:
        """Put the victim's reflected buff onto whoever hit it.

        Electro Giant's is a stun, which makes attacking it a cost in itself --
        the card punishes the defence rather than out-damaging it.
        """
        victim = hit.target
        spec = victim.spec
        if spec is None or not spec.reflected_attack_buff:
            return
        bspec = self._buff_spec(spec.reflected_attack_buff, spec.rarity, spec.level)
        if bspec is not None:
            self._apply_buff(
                hit.attacker, bspec, spec.reflected_attack_buff_ticks, source=victim.id
            )

    def _buff_spec(self, name: str, rarity: str, level: int) -> BuffSpec | None:
        key = f"{name}@{rarity}@{level}"
        if key not in self._buff_specs:
            self._buff_specs[key] = build_buff_spec(
                self.data, name, self.levels.get(rarity), level=level, clock=self.clock
            )
        return self._buff_specs[key]

    def _apply_buff(
        self, victim: Entity, spec: BuffSpec, duration: int, source: int = 0
    ) -> None:
        if duration <= 0:
            return
        if victim.buffs is None:
            victim.buffs = BuffState()
        victim.buffs.apply(spec, duration, source)

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
        if plan.is_scattered:
            # A volley is ten arrows to look at and one area to be hit by. Each
            # unit under it takes the listed damage once per wave, so it is
            # fired as a single shot carrying the *card's* radius rather than
            # the projectile's -- Arrows advertises 3.5 tiles and each arrow
            # splashes 1.4.
            #
            # Ten overlapping shots was the first attempt and it covered the
            # right ground for the wrong reason: anything near the centre was
            # caught by several of them and took two or three times the card's
            # damage.
            pspec = replace(pspec, radius=max(pspec.radius, milli_tiles(plan.radius)))
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
        if aspec.on_start_action:
            # The reworked spells keep everything here. AEO.Graveyard_rework
            # carries only a radius and a lifetime; its twelve skeletons, their
            # timings and their ring positions are all in the action graph.
            self.actions.start(
                aspec.on_start_action,
                ActionContext(team=team, x=x, y=y, source=effect),
                self.tick,
            )
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
            if isinstance(entity, RollingProjectile):
                # A roll hits continuously along its path rather than once at
                # the end, so it is swept every tick and only retired when it
                # runs out of range.
                finished = entity.advance()
                self._sweep_roll(entity)
                if finished:
                    if entity.pspec.spawn_character:
                        # Barbarian Barrel leaves its Barbarian where the roll
                        # stops, not where it was thrown.
                        self._spawn_units(
                            team=entity.team,
                            character=entity.pspec.spawn_character,
                            count=max(1, entity.pspec.spawn_count),
                            x=entity.x,
                            y=entity.y,
                            deploy_ticks=entity.pspec.spawn_deploy_ticks,
                            rarity="Common",
                        )
                    entity.kill()
                continue
            if entity.advance(self._entity(entity.target_id)):
                arrivals.append(entity)

        for shot in arrivals:
            self._impact(shot)
            shot.kill()

    def _sweep_roll(self, roll: RollingProjectile) -> None:
        """Damage and shove everything the roll is currently over.

        Each enemy is hit once for the whole pass. Damaging every tick it
        overlaps would scale the Log's damage with how slowly it crosses a
        unit, which is neither what the card does nor a number the data
        contains.
        """
        pspec = roll.pspec
        reach = max(pspec.roll_radius_x, pspec.roll_radius_y)
        for victim in list(self._index.near(roll.x, roll.y, reach)):
            if victim.dead or victim.team is roll.team or victim.id in roll.struck:
                continue
            if victim.kind is EntityKind.PROJECTILE or not victim.is_targetable:
                continue
            if victim.flying and not pspec.aoe_to_air:
                continue
            if not victim.flying and not pspec.aoe_to_ground:
                continue
            if not roll.covers(victim):
                continue
            roll.struck.add(victim.id)
            dealt = victim.apply_damage(
                pspec.damage_to(is_crown_tower=victim.kind is EntityKind.TOWER)
            )
            if dealt:
                self.damage_log.append(
                    DamageEvent(
                        tick=self.tick,
                        attacker_id=roll.owner_id,
                        target_id=victim.id,
                        amount=dealt,
                        lethal=victim.hitpoints <= 0,
                    )
                )
            if (
                pspec.pushback
                and victim.kind is EntityKind.TROOP
                and not self._pushback_immune(victim)
            ):
                # Knockback is along the roll, so a Log pushes a push back down
                # the lane rather than scattering it sideways. Giant, Golem,
                # P.E.K.K.A., Prince and Mega Knight carry IgnorePushback and
                # are exempted -- the Log rolling under a Golem and not
                # budging it is exactly the interaction the card is known for
                # *not* having against tanks.
                victim.y += pspec.pushback * roll.direction

    def _spawn_from_area(self, effect: AreaEffect) -> None:
        """One unit from a spawning area effect. Graveyard, essentially.

        Graveyard needs no action-graph interpreter: it is an ordinary area
        effect that reads ``SpawnCharacter: Skeleton``, ``SpawnInitialDelay:
        2200`` and ``SpawnInterval: 500`` over a 9000ms life. Skeletons land in
        an annulus 3 to 4 tiles from the centre rather than at the point cast,
        which is why the card surrounds a tower instead of stacking on it.

        The position is drawn from the battle's seeded stream, so a replay of
        the same seed puts every skeleton on the same tile.
        """
        aspec = effect.aspec
        if not aspec.spawn_character:
            return
        low = aspec.spawn_min_radius
        high = max(aspec.spawn_max_radius, low)
        rng = self.rng.stream(f"aeospawn:{effect.id}")
        if aspec.spawn_randomize and high > 0:
            span = max(1, high - low)
            radius = low + rng.below(span)
            eighth = rng.below(8)
        else:
            radius, eighth = high, effect.spawned % 8
        offset = ring_offsets(8, radius)[eighth]
        self._spawn_units(
            team=effect.team,
            character=aspec.spawn_character,
            count=max(1, aspec.spawn_count),
            x=effect.x + offset[0],
            y=effect.y + offset[1],
            deploy_ticks=aspec.spawn_deploy_ticks,
            rarity="Common",
        )

    def _strike(self, effect: AreaEffect) -> None:
        """Fire an area effect's own projectile at what is inside it.

        Lightning works this way: the cast places a short-lived effect that
        looses a bolt every 460ms, and ``HitBiggestTargets`` sends each bolt at
        the largest thing in range. That is the whole character of the card --
        it answers a tank and its support, and is wasted on a swarm.
        """
        aspec = effect.aspec
        candidates = [
            v
            for v in self._index.near(effect.x, effect.y, aspec.radius)
            if effect.affects(v)
            and distance(effect.x, effect.y, v.x, v.y) <= aspec.radius + v.collision_radius
            and v.id not in effect.struck
        ]
        if not candidates:
            # A shot that carries units is a delivery, not an attack: Royal
            # Delivery drops its Recruit on an empty tile exactly as it does on
            # an occupied one. Only payload-carrying shots fall through here --
            # Lightning with nothing to strike should still strike nothing.
            pspec = self._projectile_spec(aspec.projectile, "Common", 11)
            if pspec is not None and pspec.spawn_character and not effect.struck:
                effect.struck.add(effect.id)
                self._register(
                    Projectile(
                        pspec=pspec, team=effect.team, x=effect.x, y=effect.y,
                        target=effect, owner_id=effect.owner_id, spawn_tick=self.tick,
                    )
                )
            return
        if aspec.hit_biggest:
            # Largest by maximum hitpoints, with the id as a stable tiebreak.
            candidates.sort(key=lambda v: (-v.max_hitpoints, v.id))
        else:
            candidates.sort(key=lambda v: (distance(effect.x, effect.y, v.x, v.y), v.id))

        pspec = self._projectile_spec(aspec.projectile, "Common", 11)
        if pspec is None:
            return
        target = candidates[0]
        effect.struck.add(target.id)
        shot = Projectile(
            pspec=pspec, team=effect.team, x=effect.x, y=effect.y,
            target=target, owner_id=effect.owner_id, spawn_tick=self.tick,
        )
        self.entities.append(shot)
        self._by_id_map[shot.id] = shot

    def _impact(self, shot: Projectile) -> None:
        """Deliver a projectile's payload where it landed."""
        pspec = shot.pspec
        if pspec.rolls and not isinstance(shot, RollingProjectile):
            # The throw only chose where the roll starts. Handing off here
            # rather than detonating is what makes the Log a lane-clearing
            # spell instead of a small splash where it happened to land.
            self._register(
                RollingProjectile(
                    pspec=pspec,
                    team=shot.team,
                    x=shot.x,
                    y=shot.y,
                    direction=1 if shot.team is Team.BLUE else -1,
                    owner_id=shot.owner_id,
                    spawn_tick=self.tick,
                )
            )
            return
        if pspec.spawn_character:
            # Sixteen projectiles in the build carry units rather than damage.
            # Goblin Barrel is three Goblins and nothing else -- the shot has
            # no damage of its own -- so a projectile layer that ignored this
            # made the card a 3-elixir no-op, and did it silently.
            self._spawn_units(
                team=shot.team,
                character=pspec.spawn_character,
                count=max(1, pspec.spawn_count),
                x=shot.x,
                y=shot.y,
                radius=pspec.radius,
                deploy_ticks=pspec.spawn_deploy_ticks,
                rarity="Common",
            )
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
                if pspec.pushback and victim.kind is EntityKind.TROOP and not self._pushback_immune(victim):
                    # Fireball, Rocket and Snowball all carry a Pushback field
                    # that shoved nothing until this line existed -- the splash
                    # dealt its damage and stopped there. Radial from where the
                    # shot actually landed, the same derivation push_away uses
                    # for a death blast, so a unit caught on the rim of the
                    # blast is shoved less than one standing at its centre.
                    victim.x, victim.y = push_away(
                        (shot.x, shot.y), (victim.x, victim.y), pspec.pushback
                    )
            return

        target = self._entity(shot.target_id)
        if target is None or target.dead or target.team is shot.team:
            return  # the shot arrived at a corpse; it is simply spent
        self._deal(pspec, attacker, target)

    def _pushback_immune(self, victim: Entity) -> bool:
        """Giant, Golem, P.E.K.K.A., Prince, Mega Knight and the rest of the
        tank ladder: never displaced, by a Log, a Fireball or a boulder alike.

        ``IgnorePushback`` is the exact same flag :mod:`movement` already reads
        to keep these thirty units from being jostled by a crowd; it is not
        scoped to that one mechanism, and Bowler -- who carries it too -- is
        immune to his own boulder for the same reason a Golem is immune to a
        Log. Checking it here is what stops a Log knocking a P.E.K.K.A. off her
        line, which never happens in the real game.

        A buff can grant the same immunity temporarily (``BuffState.
        ignores_pushback``), which is why this is a method rather than a plain
        attribute read.
        """
        spec = victim.spec
        if spec is not None and spec.ignore_pushback:
            return True
        return victim.buffs is not None and victim.buffs.ignores_pushback()

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
        if pspec.target_buff and not victim.dead:
            # Ice Spirit's whole card is this field: TargetBuff = Freeze on its
            # own attack projectile, and nothing ever read it, so the freeze
            # a real Ice Spirit leaves behind was silently missing -- only the
            # 43 damage landed. The same field carries Lightning's stun,
            # Snowball's slow and Ice Wizard's, so this is one fix for all of
            # them rather than one per card.
            bspec = self._buff_spec(pspec.target_buff, "Common", 11)
            if bspec is not None:
                self._apply_buff(
                    victim, bspec, pspec.buff_ticks,
                    source=attacker.id if attacker is not None else 0,
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
            if entity.tick_spawn():
                self._spawn_from_area(entity)
            if not entity.tick():
                continue
            aspec = entity.aspec
            if aspec.projectile:
                self._strike(entity)
                continue
            for victim in list(self._index.near(entity.x, entity.y, aspec.radius)):
                if not entity.affects(victim):
                    continue
                if distance(entity.x, entity.y, victim.x, victim.y) > aspec.radius + victim.collision_radius:
                    continue
                if aspec.on_hit_action:
                    # Per affected entity, not once at the centre. Clone acts on
                    # each friendly troop it touches; running it at the point
                    # cast would duplicate nothing at all.
                    self.actions.run(
                        aspec.on_hit_action,
                        ActionContext(
                            team=entity.team, x=victim.x, y=victim.y, source=victim,
                            variables={
                                "buff_ticks": aspec.buff_ticks or aspec.life_ticks,
                                "owner": entity.id,
                            },
                        ),
                        self.tick,
                    )
                if aspec.buff:
                    # Keyed by the effect's own id, so re-touching a unit every
                    # scan refreshes its status instead of stacking a new copy
                    # -- BuffNumber is 1 on every area effect in the build. Two
                    # separate clouds remain two sources and do stack.
                    bspec = self._buff_spec(aspec.buff, "Common", 11)
                    if bspec is not None:
                        self._apply_buff(
                            victim,
                            bspec,
                            aspec.buff_ticks or aspec.life_ticks,
                            source=entity.id,
                        )
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

    def _phase_pull_units(self) -> None:
        """Drag units toward whatever is attracting them. Tornado, essentially.

        Its own phase, ahead of movement, because a pull is a force applied
        *to* a unit rather than a decision made *by* one. That distinction is
        the whole card: a Tornado has to move a unit that is standing still to
        attack, and a unit that is frozen solid, and a unit whose own speed is
        zero. Folding this into :meth:`_phase_move_units` would skip every one
        of those cases, since that phase returns early for all of them -- and
        dragging a committed push off the bridge and into the King Tower is
        precisely what Tornado is for.

        Buildings and towers are not pulled: nothing in the game drags a
        Cannon, and the data has no notion of moving one.
        """
        for entity in self.entities:
            if entity.dead or entity.buffs is None or entity.is_deploying:
                continue
            if entity.kind is not EntityKind.TROOP:
                continue
            for source_id, tiles_per_minute in entity.buffs.attract_sources():
                source = self._entity(source_id)
                if source is None or source.dead:
                    continue
                step = tiles_per_minute * SUBTILES_PER_TILE // (60 * self.clock.ticks_per_second)
                if step <= 0:
                    continue
                x, y = step_towards((entity.x, entity.y), (source.x, source.y), step)
                # Deliberately not routed and not walkability-checked: this is a
                # displacement, the same as pushback, and a tornado over the
                # river really does hold units above it.
                entity.x, entity.y = x, y

    def _accumulate_charge(self, entity: Entity, x: int, y: int) -> None:
        """Count ground covered toward a charge.

        Measured from where the unit actually ends up rather than from the step
        it intended, so a Prince shoved sideways by a crowd or sliding along a
        river bank builds his charge from the distance he really travelled.
        """
        spec = entity.spec
        if spec is None or not spec.charge_range:
            return
        moved = distance(entity.x, entity.y, x, y)
        if moved:
            self._charge[entity.id] = self._charge.get(entity.id, 0) + moved

    def _is_charged(self, entity: Entity, spec: UnitSpec) -> bool:
        return self._charge.get(entity.id, 0) >= spec.charge_range

    def _refresh_occupancy(self) -> None:
        """Tell the path grid where the buildings are.

        Rebuilt from the live list rather than maintained incrementally: a
        building can leave by death, by lifetime, or by being cloned away, and
        an incremental map would need every one of those paths to remember to
        update it. The grid only bumps its version when the result differs, so
        rebuilding an unchanged map costs a comparison and invalidates nothing.
        """
        # Buildings are static once placed, so the map only changes when one
        # appears or dies. A signature of what is standing is far cheaper than
        # rebuilding the map, and rebuilding it every tick cost 40% of the
        # engine's throughput for a result that was identical 99% of the time.
        half = SUBTILES_PER_TILE // 2
        standing = tuple(
            (e.id, e.x // half, e.y // half)
            for e in self.entities
            if not e.dead and e.kind in (EntityKind.BUILDING, EntityKind.TOWER)
        )
        if standing == self._occupancy_signature:
            return
        self._occupancy_signature = standing

        cost = self.path_grid.costs["building"]
        occupied: dict[int, int] = {}
        for entity in self.entities:
            if entity.dead or entity.kind not in (EntityKind.BUILDING, EntityKind.TOWER):
                continue
            radius = entity.collision_radius
            cx, cy = entity.x // half, entity.y // half
            reach = max(0, radius // half)
            for oy in range(-reach, reach + 1):
                for ox in range(-reach, reach + 1):
                    occupied[self.path_grid.index_of(cx + ox, cy + oy)] = cost
        self.path_grid.set_occupancy(occupied)

    def _phase_move_units(self) -> None:
        self._refresh_occupancy()
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

            step = spec.speed_per_tick
            if spec.charge_range and self._is_charged(entity, spec):
                # A charged unit gallops. Prince's ChargeSpeedMultiplier is
                # 200, and the acceleration is half of what makes the card
                # frightening -- it closes the ground you were relying on.
                step = step * spec.charge_speed_multiplier // 100
            if entity.buffs is not None:
                # A frozen unit does not move at all; a slowed or raged one
                # moves at its adjusted rate.
                step = apply_delta(step, entity.buffs.speed_multiplier())
                if step <= 0:
                    # Stopped, so the run-up is lost. Checked here as well as
                    # in the attack phase because a charger frozen on open
                    # ground never reaches that branch at all -- it has no
                    # target in range to be attacking.
                    if spec.charge_range:
                        self._charge[entity.id] = 0
                    entity.set_state(EntityState.IDLE)
                    continue

            target = self._entity(entity.target_id)
            if target is not None and not target.dead:
                if in_attack_range(spec, entity, target):
                    # In reach: stop and fight. Units standing still to attack is
                    # what makes a push advance at its tank's pace.
                    self._routes.pop(entity.id, None)
                    continue
                goal = (target.x, target.y)
                blocked = line_blocked(
                    self.path_grid, (entity.x, entity.y), goal, flying=entity.flying
                )
                if entity.flying or (
                    not blocked and not crosses_river(self.arena, entity.y, target.y)
                ):
                    # Same side of the water (or airborne): walk straight at it.
                    # Building a route here would mean rebuilding it every tick,
                    # since the destination moves.
                    self._routes.pop(entity.id, None)
                    self._place(
                        entity,
                        *step_towards((entity.x, entity.y), goal, step),
                    )
                else:
                    # Across the water: this genuinely needs a bridge, and the
                    # plan survives until the crossing is done.
                    route = self._routes.get(entity.id)
                    if route is None or route.finished:
                        route = route_to(
                            self.arena, (entity.x, entity.y), goal,
                            flying=entity.flying, grid=self.path_grid,
                        )
                        self._routes[entity.id] = route
                    self._place(
                        entity,
                        *route.advance((entity.x, entity.y), step),
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
                    self.arena, (entity.x, entity.y), goal,
                    flying=entity.flying, grid=self.path_grid,
                )
                self._routes[entity.id] = route
            self._place(entity, *route.advance((entity.x, entity.y), step))
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
            self._accumulate_charge(entity, x, y)
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
                self._last_attack.pop(entity.id, None)
                self._spawn_timers.pop(entity.id, None)
                self._spawn_children.pop(entity.id, None)
                self._charge.pop(entity.id, None)
                self._hit_counts.pop(entity.id, None)
                died = True
                if entity.kind is EntityKind.TOWER:
                    self.players[entity.team.opponent].crowns += 1
                else:
                    # Towers leave nothing behind; everything else might.
                    self._resolve_death_payload(entity)
        if died:
            alive: list[Entity] = []
            for entity in self.entities:
                if entity.dead:
                    self.graveyard.append(entity)
                else:
                    alive.append(entity)
            self.entities = alive

    def _resolve_death_payload(self, entity: Entity) -> None:
        """Everything a unit leaves behind when it dies.

        Three separate things, and a card can carry any combination. Golem has
        the first two, Ice Golem the second and third, a hut only the first.

        This is most of what you pay for on the expensive cards. A Golem that
        did not split would be a worse Giant, and killing one would end a push
        rather than begin the second half of it -- so a death that produces
        nothing is not a neutral simplification, it deletes the card.
        """
        # A buff can carry a death spawn of its own, and it belongs to
        # whoever applied the buff rather than to whoever died -- which is the
        # whole of Goblin Curse: what dies under it comes back on your side.
        if entity.buffs is not None:
            for character, count, to_applier, source_id in entity.buffs.death_spawns():
                owner = self._entity(source_id)
                team = entity.team
                if to_applier:
                    team = owner.team if owner is not None else entity.team.opponent
                self._spawn_units(
                    team=team, character=character, count=count,
                    x=entity.x, y=entity.y, rarity="Common",
                )

        spec = entity.spec
        if spec is None:
            return

        if spec.death_damage and spec.death_damage_radius:
            self._death_blast(entity, spec)

        if spec.death_area_effect:
            self._place_area(
                entity.team, spec.death_area_effect, spec.rarity, spec.level,
                entity.x, entity.y, owner_id=entity.id,
            )

        for character in (spec.death_spawn_character, spec.death_spawn_character2):
            if not character:
                continue
            self._spawn_units(
                team=entity.team,
                character=character,
                count=max(1, spec.death_spawn_count),
                x=entity.x,
                y=entity.y,
                radius=spec.death_spawn_radius,
                deploy_ticks=spec.death_spawn_deploy_ticks,
                rarity=spec.rarity,
            )

    def _death_blast(self, entity: Entity, spec: UnitSpec) -> None:
        """A unit's parting explosion.

        Hits both sides. Golem's 88 landing on your own support is a real cost
        of the card, and filtering it to enemies only would quietly make every
        big death spawn strictly better than it is.
        """
        for victim in list(self._index.near(entity.x, entity.y, spec.death_damage_radius)):
            if victim.dead or victim.id == entity.id:
                continue
            if victim.kind in UNTARGETABLE_KINDS or not victim.is_targetable:
                continue
            if distance(entity.x, entity.y, victim.x, victim.y) > (
                spec.death_damage_radius + victim.collision_radius
            ):
                continue
            dealt = victim.apply_damage(
                spec.death_damage
                * (100 + spec.crown_tower_damage_percent) // 100
                if victim.kind is EntityKind.TOWER and spec.crown_tower_damage_percent
                else spec.death_damage
            )
            if dealt:
                self.damage_log.append(
                    DamageEvent(
                        tick=self.tick, attacker_id=entity.id, target_id=victim.id,
                        amount=dealt, lethal=victim.hitpoints <= 0,
                    )
                )
            if spec.death_pushback and victim.kind is EntityKind.TROOP:
                victim.x, victim.y = push_away(
                    (entity.x, entity.y), (victim.x, victim.y), spec.death_pushback
                )

    def _spawn_units(
        self,
        *,
        team: Team,
        character: str,
        count: int,
        x: int,
        y: int,
        radius: int = 0,
        deploy_ticks: int = 0,
        stagger_ticks: int = 0,
        rarity: str = "Common",
    ) -> list[Entity]:
        """Put ``count`` copies of one character on the board around a point.

        Shared by death spawns and by the huts, because they are the same
        operation: the only differences are where the point comes from and how
        long the arrivals are staggered.

        A group with no radius of its own is packed rather than stacked. Units
        on one exact point are perfectly overlapped, so any splash catches all
        of them -- a Tombstone's four Skeletons would die to a single Zap.
        """
        try:
            spec = self._spec(character, rarity=rarity)
        except Exception:
            # A renamed or event-only character is not worth aborting a death
            # over; it simply leaves nothing behind.
            return []
        if count > 1:
            offsets = (
                ring_offsets(count, radius)
                if radius
                else pack_offsets(count, spec.collision_radius)
            )
        else:
            offsets = ((0, 0),)

        spawned: list[Entity] = []
        for index in range(count):
            ox, oy = offsets[index] if index < len(offsets) else (0, 0)
            px, py = self._settle(x + ox, y + oy, flying=spec.flying)
            unit = Entity(
                kind=spec.kind,
                team=team,
                x=px,
                y=py,
                hitpoints=spec.hitpoints,
                spec=spec,
                spawn_tick=self.tick,
                deploy_ticks=(deploy_ticks or spec.deploy_ticks) + index * stagger_ticks,
                collision_radius=spec.collision_radius,
                mass=spec.mass,
                flying=spec.flying,
                shield=spec.shield_hitpoints,
                lifetime_ticks=spec.lifetime_ticks,
            )
            self._register(unit)
            self._begin_actions(unit)
            spawned.append(unit)
        return spawned

    def _place_area_from_action(
        self, team: Team, name: str, x: int, y: int, source: Entity | None
    ) -> None:
        """Place an area effect on behalf of an action node."""
        spec = source.spec if source is not None else None
        self._place_area(
            team, name,
            spec.rarity if spec is not None else "Common",
            spec.level if spec is not None else self.config.level,
            x, y,
            owner_id=source.id if source is not None else 0,
        )

    def _apply_buff_from_action(self, ctx, name: str, row) -> None:
        """Put a named buff on whatever the action is running against.

        The duration is the one puzzle. A buff carries no lifetime of its own
        -- that belongs to whatever applies it -- and these actions carry a
        ``SpawnTime`` that is a spawn delay rather than a duration. So the
        effect that fired the action passes its own remaining life down, which
        is right for the case this exists for: Goblin Curse's cloud re-applies
        every 50ms for six seconds, so each application only has to outlive the
        next one.
        """
        target = ctx.source
        if target is None or target.dead:
            return
        spec = target.spec
        bspec = self._buff_spec(
            name,
            spec.rarity if spec is not None else "Common",
            spec.level if spec is not None else self.config.level,
        )
        if bspec is None:
            return
        duration = ctx.variables.get("buff_ticks") or self.clock.ticks(row.get("SpawnTime"))
        self._apply_buff(target, bspec, max(1, int(duration)), source=ctx.variables.get("owner", 0))

    def _count_living(self, team: Team, names: set[str]) -> int:
        """How many living units of the given names a team has on the board."""
        return sum(
            1
            for entity in self.entities
            if not entity.dead
            and entity.team is team
            and entity.spec is not None
            and entity.spec.name in names
        )

    def _clone_entity(self, source: Entity) -> Entity | None:
        """Duplicate a troop, the way the Clone spell does.

        A clone is the same unit with one hitpoint. Full damage, full speed,
        the original's shield if it had one -- which is the whole reason to
        clone a Dark Prince, and why the spell is a damage multiplier rather
        than a health one.

        The single hitpoint is the one figure not in this build: not in
        ``ACTION.CloneAction``, not in ``BUFF.Clone``, and not in any
        ``CLONE_*`` global. It is applied as the card's known behaviour and
        recorded as an open anchor rather than passed off as data.
        """
        spec = source.spec
        if spec is None:
            return None
        globals_map = self.data.globals_map()
        offset_x = _int_global(globals_map, "CLONE_DISTANCE_X", 0) * 18
        offset_y = _int_global(globals_map, "CLONE_DISTANCE_Y", 250) * 18
        keep_shield = globals_map.get("CLONE_PRESERVE_SHIELD") is True

        clone = Entity(
            kind=spec.kind,
            team=source.team,
            x=source.x + offset_x,
            y=source.y + offset_y * (1 if source.team is Team.BLUE else -1),
            hitpoints=CLONE_HITPOINTS,
            spec=spec,
            spawn_tick=self.tick,
            collision_radius=source.collision_radius,
            mass=source.mass,
            flying=source.flying,
            shield=source.shield if keep_shield else 0,
            lifetime_ticks=spec.lifetime_ticks,
        )
        clone.max_hitpoints = CLONE_HITPOINTS
        clone.is_clone = True
        self._register(clone)
        return clone

    def _settle(self, x: int, y: int, *, flying: bool) -> tuple[int, int]:
        """Nudge a spawn point to somewhere the unit can actually stand.

        Movement has always guarded this; spawning never did. Every offset that
        places a unit relative to a point can push it off legal ground -- a
        swarm's ring, a death spawn's radius, a Graveyard skeleton's annulus --
        and a soak run found troops standing in the river in 2.5% of matches
        because of it.

        Searched outward in half-tile rings and capped: a point with nothing
        legal near it returns unchanged, because a unit slightly out of place
        is a smaller wrong than one teleported across the board.
        """
        if self.arena.is_walkable(x, y, flying=flying):
            return x, y
        half = SUBTILES_PER_TILE // 2
        for ring in range(1, 5):
            for dx, dy in ring_offsets(8, ring * half):
                if self.arena.is_walkable(x + dx, y + dy, flying=flying):
                    return x + dx, y + dy
        return x, y

    def _begin_actions(self, entity: Entity) -> None:
        """Fire an entity's ``OnStartingAction``, if it has one."""
        spec = entity.spec
        if spec is None or not spec.on_starting_action:
            return
        self.actions.start(
            spec.on_starting_action,
            ActionContext(team=entity.team, x=entity.x, y=entity.y, source=entity),
            self.tick,
        )

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

    def fallen_enemy_towers(self, team: Team) -> frozenset[TowerPlacement]:
        """``team``'s opponent's destroyed Princess Towers.

        This is what ``Arena.can_deploy`` needs to know its deploy zone has
        expanded -- see that method's docstring for the rule. Public (unlike
        ``_king``) because both ``play_card`` and the RL action-mask builder
        in :mod:`cr_sim.api.encoding` need the same answer, and it must agree
        between them or an agent could be trained against legality the human
        path does not actually honour.

        Only Princess Towers matter: a destroyed King Tower ends the match in
        ``_phase_check_victory`` before an expanded zone would ever be used.
        Read from ``self._towers`` rather than the live entity list for the
        same reason ``_king`` is -- a destroyed tower is retired to the
        graveyard, not left in place with zero hitpoints.
        """
        return frozenset(
            TowerPlacement(t.spec.name, t.team, t.x, t.y)
            for t in self._towers[team.opponent]
            if t.dead and "King" not in getattr(t.spec, "name", "")
        )

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
