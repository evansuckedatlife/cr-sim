"""What the agent is actually paid for.

Crown difference is the true objective and a terrible training signal on its
own. A 120-second match usually ends 0-0: neither side takes a tower, so most
episodes produce a reward of exactly zero and the policy learns nothing from
them. Measured on this engine, uniform-random play wins 29% of matches against
an opponent that never plays a card -- not because random play is good, but
because the remaining 71% end level and count as neither win nor loss.

So the reward is built from five things a player would actually name as
progress, all of them observable between crowns:

``tower_damage``
    Damage put on enemy towers. The thing crowns are made of, available
    continuously rather than in three lumps.
``own_tower_hp``
    Damage taken. Separate from the above rather than netted, because
    defending well and attacking well are different skills and a single
    difference term cannot distinguish "traded evenly" from "did nothing".
``elixir_trade``
    Enemy elixir destroyed minus your own elixir lost. The currency the whole
    game is played in.
``counterpush``
    Elixir still standing. Winning a defence with units left over is worth
    more than winning it cleanly, because those units are the next attack.
``kite``
    Time spent holding an enemy unit off its target with something cheaper --
    an Ice Golem walking a P.E.K.K.A into the middle of the board. Counted
    only when the kiting unit costs less than what it is holding, because
    that price difference is the whole point; a Knight holding a Skeleton is
    just a block.

**Everything is a potential, and the reward is its change.** Each term
contributes to a single running score, and a step is paid the difference in
that score. This is potential-based shaping: summed over an episode the
shaping terms telescope to a constant, so they cannot change which policy is
optimal -- they only make the gradient dense enough to find it. Paying a term
directly instead would let the agent farm it, and the obvious farm here is
elixir: a term that rewards having units on the board, without the matching
negative when they die, is a reward for dumping your hand.

**Values are per unit, not per card.** Skeletons cost one elixir for four
bodies, so each body is worth a quarter. Without that a Skeleton trades as
though it cost the same as a P.E.K.K.A, and every kite and elixir trade term
reads backwards for exactly the cards those terms exist to reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..data.cards import CardRegistry
from ..engine.battle import Battle
from ..engine.entity import Entity, EntityKind, Team

__all__ = ["RewardWeights", "RewardTracker", "unit_elixir_values"]


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """How much each term contributes to the running score.

    Tower terms sit at 1.0 so they are commensurate with a crown, which is also
    1.0. The rest are deliberately smaller: they are means, not ends, and an
    agent that maximises elixir trades while never touching a tower has learned
    the wrong game. They are large enough to shape and too small to chase.
    """

    tower_damage: float = 1.0
    own_tower_hp: float = 1.0
    elixir_trade: float = 0.3
    counterpush: float = 0.2
    kite: float = 0.1
    #: Crowns are the real objective and are not shaping, so this is not
    #: reduced. Kept here so the whole reward is described in one place.
    crowns: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def unit_elixir_values(registry: CardRegistry) -> dict[str, float]:
    """Elixir value of one body of each character the game can deploy.

    Divided by the summon count, so a Skeleton is 0.25 rather than 1.0. Where
    two cards summon the same character the cheaper per-body figure wins, which
    keeps a unit from being valued by the most expensive way of obtaining it.
    """
    values: dict[str, float] = {}
    for card in registry.standard():
        summons = card.summons()
        total = sum(count for _, count in summons) or 1
        for character, _count in summons:
            per_body = card.mana_cost / total
            existing = values.get(character)
            if existing is None or per_body < existing:
                values[character] = per_body
    return values


@dataclass(slots=True)
class _Snapshot:
    """The parts of a battle the score is computed from."""

    own_tower_hp: float = 0.0
    enemy_tower_hp: float = 0.0
    own_tower_max: float = 0.0
    enemy_tower_max: float = 0.0
    own_units: float = 0.0


class RewardTracker:
    """Turns battle state into a reward, one team's point of view.

    Reads the battle rather than instrumenting it: nothing in the engine knows
    this exists. That matters because the reward is a training choice and the
    simulator is supposed to be a statement about Clash Royale -- an engine
    carrying reward bookkeeping would be a worse model of the game and a
    harder thing to verify.
    """

    __slots__ = (
        "team", "weights", "_values", "_seen", "_lost", "_destroyed",
        "_kite_ticks", "_previous", "_terms",
    )

    def __init__(
        self,
        team: Team,
        registry: CardRegistry,
        weights: RewardWeights | None = None,
    ) -> None:
        self.team = team
        self.weights = weights or RewardWeights()
        self._values = unit_elixir_values(registry)
        #: Units already counted, so a death is attributed once. Ids are unique
        #: for a battle's lifetime, so this cannot collide across respawns.
        self._seen: dict[int, tuple[Team, float]] = {}
        self._lost = 0.0
        self._destroyed = 0.0
        self._kite_ticks = 0.0
        self._previous = 0.0
        self._terms: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def reset(self, battle: Battle) -> None:
        self._seen.clear()
        self._lost = 0.0
        self._destroyed = 0.0
        self._kite_ticks = 0.0
        self._terms = {}
        self._previous = self.score(battle)

    def step(self, battle: Battle, elapsed_ticks: int) -> float:
        """Reward for the ticks just simulated, as the change in the score."""
        self._observe(battle, elapsed_ticks)
        current = self.score(battle)
        reward = current - self._previous
        self._previous = current
        return reward

    @property
    def terms(self) -> dict[str, float]:
        """Each term's current contribution, for logging.

        Reported separately because a single reward number cannot say whether
        the agent is learning to attack, to trade, or to farm one term while
        ignoring the game.
        """
        return dict(self._terms)

    # -- the score ---------------------------------------------------------

    def score(self, battle: Battle) -> float:
        """The running potential. A step is paid the change in this."""
        weights = self.weights
        opponent = self.team.opponent
        snapshot = self._snapshot(battle)

        own_fraction = (
            snapshot.own_tower_hp / snapshot.own_tower_max if snapshot.own_tower_max else 0.0
        )
        enemy_damage = (
            1.0 - snapshot.enemy_tower_hp / snapshot.enemy_tower_max
            if snapshot.enemy_tower_max
            else 0.0
        )
        crowns = battle.players[self.team].crowns - battle.players[opponent].crowns
        # Normalised by 10 elixir, a full bar, so a term measured in elixir is
        # commensurate with one measured as a fraction of a tower.
        trade = (self._destroyed - self._lost) / 10.0
        counterpush = snapshot.own_units / 10.0
        kite = self._kite_ticks / 10.0

        self._terms = {
            "crowns": weights.crowns * crowns,
            "tower_damage": weights.tower_damage * enemy_damage,
            "own_tower_hp": weights.own_tower_hp * own_fraction,
            "elixir_trade": weights.elixir_trade * trade,
            "counterpush": weights.counterpush * counterpush,
            "kite": weights.kite * kite,
        }
        return sum(self._terms.values())

    # -- observation -------------------------------------------------------

    def _snapshot(self, battle: Battle) -> _Snapshot:
        snapshot = _Snapshot()
        for tower in battle._towers[self.team]:
            snapshot.own_tower_max += tower.max_hitpoints
            snapshot.own_tower_hp += max(0, tower.hitpoints)
        for tower in battle._towers[self.team.opponent]:
            snapshot.enemy_tower_max += tower.max_hitpoints
            snapshot.enemy_tower_hp += max(0, tower.hitpoints)
        for entity in battle.entities:
            if entity.dead or entity.team is not self.team:
                continue
            if entity.kind is EntityKind.TROOP or entity.kind is EntityKind.BUILDING:
                snapshot.own_units += self._value(entity)
        return snapshot

    def _observe(self, battle: Battle, elapsed_ticks: int) -> None:
        """Update the cumulative terms: deaths, and time spent kiting."""
        for entity in battle.entities:
            if entity.kind in (EntityKind.TROOP, EntityKind.BUILDING):
                self._seen.setdefault(entity.id, (entity.team, self._value(entity)))

        # The graveyard is the engine's own record of what has died, so a death
        # is never missed by a unit that came and went between two observations.
        for entity in battle.graveyard:
            record = self._seen.pop(entity.id, None)
            if record is None:
                continue
            team, value = record
            if team is self.team:
                self._lost += value
            else:
                self._destroyed += value

        self._kite_ticks += self._kiting(battle) * elapsed_ticks / 60.0

    def _kiting(self, battle: Battle) -> float:
        """Elixir-weighted count of enemies currently being held by something cheaper.

        An enemy targeting one of your troops is not attacking your tower. That
        is only worth paying for when the thing holding it is cheaper than it
        is -- an Ice Golem on a P.E.K.K.A is a kite, a Knight on a Skeleton is
        just a fight, and rewarding the second teaches the agent to trade down.
        """
        held = 0.0
        for entity in battle.entities:
            if entity.dead or entity.team is self.team:
                continue
            if entity.kind is not EntityKind.TROOP:
                continue
            target = battle._entity(entity.target_id)
            if target is None or target.dead or target.team is not self.team:
                continue
            if target.kind is EntityKind.TOWER:
                # Doing exactly what it wanted to do. Not a kite.
                continue
            enemy_value = self._value(entity)
            if self._value(target) < enemy_value:
                held += enemy_value
        return held

    def _value(self, entity: Entity) -> float:
        spec = entity.spec
        if spec is None:
            return 0.0
        return self._values.get(spec.name, 1.0)


def describe(tracker: RewardTracker) -> dict[str, Any]:
    """Flat, loggable view of a tracker's current terms."""
    return {f"reward/{name}": value for name, value in tracker.terms.items()}
