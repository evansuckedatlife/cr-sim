"""A live match a person can play against a policy.

Everything else in this package runs battles as fast as it can and reads the
result afterwards. This one has to run at the speed of a clock, accept input at
any moment, and be describable to a browser several times a second -- which
makes it a different shape of problem from either the replay viewer or the
training environment, even though it sits on the same engine.

Two decisions shape it.

**The battle runs on the server and the browser only draws.** The alternative
-- shipping the state and simulating in JavaScript -- would mean a second
implementation of the engine, and a second implementation is a second set of
answers to every question the first one spent this project settling. The page
gets positions and hitpoints; it decides nothing.

**Time advances in whole engine ticks driven by a real clock.** The session is
told how much wall-clock time has passed and converts it to ticks, rather than
stepping once per animation frame. A dropped frame or a slow request then costs
smoothness, never determinism: the same inputs at the same ticks produce the
same battle, which is what makes a match reproducible from its command log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..data.cards import CardRegistry, build_card_registry
from ..data.leveling import LevelTable, build_level_table
from ..data.source import LogicData
from ..engine.battle import Battle, BattleConfig
from ..engine.entity import EntityKind, Team
from ..engine.fixed import to_tiles

__all__ = ["PlaySession", "SessionConfig", "AiController", "random_controller"]

#: Real seconds a session will simulate in one catch-up. A tab left in the
#: background stops asking for frames, and without a ceiling the first request
#: after it returns would try to simulate every tick it missed at once and
#: freeze for as long as the tab was hidden.
MAX_CATCHUP_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class SessionConfig:
    human_deck: tuple[str, ...]
    ai_deck: tuple[str, ...]
    seed: int = 0
    ticks_per_second: int = 60
    level: int = 11
    tower_level: int = 11
    #: How often the AI is allowed to act, in ticks. It thinks on a cadence
    #: rather than every tick for the same reason the training environment
    #: does: there is nothing to decide between elixir arriving, and an
    #: opponent that re-evaluates 60 times a second is not a better opponent,
    #: only a more expensive one.
    ai_interval_ticks: int = 30


#: Given a battle and a team, return ``(card, x, y)`` in subtiles, or None.
AiController = Callable[[Battle, Team], "tuple[str, int, int] | None"]


def random_controller(seed: int = 0) -> AiController:
    """An opponent that spends its elixir on legal placements.

    The fallback when no trained policy is supplied. Weak, but it plays cards,
    which makes it a far better sparring partner than an idle opponent: most of
    what a person wants to see -- units meeting in the lane, a tower being
    defended -- does not happen at all against a side that never deploys.
    """
    from ..engine.rng import Rng

    rng = Rng(seed).stream("opponent")

    def controller(battle: Battle, team: Team) -> tuple[str, int, int] | None:
        player = battle.players[team]
        affordable = [
            name for name in player.hand
            if (card := battle.registry.get(name)) is not None
            and player.elixir.can_afford(card.mana_cost)
        ]
        if not affordable:
            return None
        card = affordable[rng.below(len(affordable))]
        lane = (3.5, 14.5)[rng.below(2)]
        depth = 20.0 + rng.below(6) if team is Team.RED else 11.0 - rng.below(6)
        from ..engine.fixed import tiles

        return card, tiles(lane), tiles(depth)

    return controller


@dataclass
class PlaySession:
    """One live match, with a human on one side and a controller on the other."""

    data: LogicData
    levels: LevelTable
    registry: CardRegistry
    config: SessionConfig
    controller: AiController | None = None
    human_team: Team = Team.BLUE

    battle: Battle = field(init=False)
    _last_wall: float = field(init=False, default=0.0)
    _tick_debt: float = field(init=False, default=0.0)
    _next_ai_tick: int = field(init=False, default=0)
    _log: list[dict[str, Any]] = field(init=False, default_factory=list)
    paused: bool = field(init=False, default=False)
    speed: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, config: SessionConfig | None = None) -> None:
        if config is not None:
            self.config = config
        blue = self.config.human_deck if self.human_team is Team.BLUE else self.config.ai_deck
        red = self.config.ai_deck if self.human_team is Team.BLUE else self.config.human_deck
        self.battle = Battle(
            self.data, self.levels, self.registry,
            BattleConfig(
                seed=self.config.seed,
                ticks_per_second=self.config.ticks_per_second,
                blue_deck=blue,
                red_deck=red,
                level=self.config.level,
                tower_level=self.config.tower_level,
            ),
        )
        self._last_wall = time.monotonic()
        self._tick_debt = 0.0
        self._next_ai_tick = 0
        self._log.clear()
        self.paused = False

    @property
    def ai_team(self) -> Team:
        return self.human_team.opponent

    # -- input -------------------------------------------------------------

    def play(self, card: str, tile_x: float, tile_y: float) -> dict[str, Any]:
        """Place a card for the human side. Returns why it failed, if it did.

        The reason matters: a card that silently does not appear reads as a bug
        to whoever pressed the button, and there are four quite different
        reasons it can happen.
        """
        from ..engine.fixed import tiles

        player = self.battle.players[self.human_team]
        if self.battle.finished:
            return {"ok": False, "reason": "the match is over"}
        if card not in player.hand:
            return {"ok": False, "reason": f"{card} is not in hand"}
        entry = self.registry.get(card)
        if entry is None:
            return {"ok": False, "reason": f"unknown card {card}"}
        if not player.elixir.can_afford(entry.mana_cost):
            return {"ok": False, "reason": f"needs {entry.mana_cost} elixir"}

        x, y = tiles(tile_x), tiles(tile_y)
        if not self.battle.play_card(self.human_team, card, x, y):
            return {"ok": False, "reason": "cannot be placed there"}
        self._log.append({"tick": self.battle.tick, "team": self.human_team.name,
                          "card": card, "x": tile_x, "y": tile_y})
        return {"ok": True}

    # -- time --------------------------------------------------------------

    def advance(self, now: float | None = None) -> int:
        """Simulate however many ticks real time has earned. Returns the count."""
        now = time.monotonic() if now is None else now
        elapsed = now - self._last_wall
        self._last_wall = now
        if self.paused or self.battle.finished:
            return 0

        elapsed = min(elapsed, MAX_CATCHUP_SECONDS)
        self._tick_debt += elapsed * self.config.ticks_per_second * self.speed
        ticks = int(self._tick_debt)
        self._tick_debt -= ticks

        for _ in range(ticks):
            if self.battle.finished:
                break
            self._think()
            self.battle.step()
        return ticks

    def _think(self) -> None:
        """Give the controller its turn, on its own cadence."""
        if self.controller is None or self.battle.tick < self._next_ai_tick:
            return
        self._next_ai_tick = self.battle.tick + self.config.ai_interval_ticks
        choice = self.controller(self.battle, self.ai_team)
        if choice is None:
            return
        card, x, y = choice
        if self.battle.play_card(self.ai_team, card, x, y):
            self._log.append({"tick": self.battle.tick, "team": self.ai_team.name,
                              "card": card, "x": to_tiles(x), "y": to_tiles(y)})

    # -- output ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs to draw one frame.

        Deliberately flat and small. This is serialised several times a second,
        so it carries what is drawn and nothing else -- no specs, no history,
        nothing the client would have to be taught to interpret.
        """
        battle = self.battle
        entities = []
        for entity in battle.entities:
            if entity.dead or entity.kind is EntityKind.PROJECTILE:
                continue
            spec = entity.spec
            entities.append({
                "id": entity.id,
                "n": spec.name if spec is not None else entity.kind.name,
                "t": int(entity.team),
                "k": int(entity.kind),
                "x": round(to_tiles(entity.x), 3),
                "y": round(to_tiles(entity.y), 3),
                "hp": max(0, entity.hitpoints),
                "max": max(1, entity.max_hitpoints),
                "sh": entity.shield,
                "d": entity.is_deploying,
                "f": entity.flying,
            })

        return {
            "tick": battle.tick,
            "seconds": round(battle.tick / self.config.ticks_per_second, 2),
            "finished": battle.finished,
            "paused": self.paused,
            "speed": self.speed,
            "result": self._result(),
            "entities": entities,
            "you": self._side(self.human_team),
            "them": self._side(self.ai_team),
        }

    def _side(self, team: Team) -> dict[str, Any]:
        player = self.battle.players[team]
        return {
            "team": int(team),
            # `exact` is the display value; `units` is what a spend checks.
            "elixir": round(player.elixir.exact, 2),
            "crowns": player.crowns,
            "hand": [self._card(name) for name in player.hand],
            "next": self._card(player.next_card) if player.next_card else None,
            "towers": [
                {
                    "n": t.spec.name if t.spec else "Tower",
                    "hp": max(0, t.hitpoints),
                    "max": max(1, t.max_hitpoints),
                    "x": round(to_tiles(t.x), 2),
                    "y": round(to_tiles(t.y), 2),
                }
                for t in self.battle._towers[team]
            ],
        }

    def _card(self, name: str) -> dict[str, Any]:
        entry = self.registry.get(name)
        return {
            "name": name,
            "cost": entry.mana_cost if entry is not None else 0,
            "kind": entry.kind.value if entry is not None else "troop",
        }

    def _result(self) -> dict[str, Any] | None:
        result = self.battle.result
        if result is None:
            return None
        you = self.battle.players[self.human_team].crowns
        them = self.battle.players[self.ai_team].crowns
        return {
            "reason": result.reason,
            "you": you,
            "them": them,
            "outcome": "win" if you > them else "loss" if them > you else "draw",
        }

    @property
    def commands(self) -> list[dict[str, Any]]:
        """Every card played, in order. A match is replayable from this."""
        return list(self._log)


def build_session(
    build: Any,
    human_deck: Sequence[str],
    ai_deck: Sequence[str],
    *,
    controller: AiController | None = None,
    seed: int = 0,
) -> PlaySession:
    """Load a build and start a session against ``controller``."""
    data = LogicData.load(build)
    return PlaySession(
        data=data,
        levels=build_level_table(data),
        registry=build_card_registry(data),
        config=SessionConfig(human_deck=tuple(human_deck), ai_deck=tuple(ai_deck), seed=seed),
        controller=controller or random_controller(seed),
    )
