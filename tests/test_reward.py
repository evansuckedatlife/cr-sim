"""The five-term training reward.

Crown difference is the true objective and a useless training signal by itself:
a 120-second match usually ends 0-0, so most episodes pay exactly zero. These
terms are the things a player would name as progress in between crowns.

The failures worth guarding against are all quiet ones. A reward term that
never fires looks identical to one that fires and does not help. A term that
can be farmed produces an agent that maximises it and loses every match. And a
unit valued per card rather than per body makes a Skeleton trade like a
P.E.K.K.A, which reads backwards for exactly the cards these terms exist to
reward.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_sim.api.env import CRSimEnv
from cr_sim.api.reward import RewardTracker, RewardWeights, unit_elixir_values
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons", "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world):
    data, levels, registry = world
    return Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=DECK, red_deck=DECK),
    )


def _spawn(battle, world, unit, team, x, y, *, rarity="Common"):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, unit, level=11, rarity=rarity, clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y), hitpoints=spec.hitpoints,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = entity.hitpoints
    entity.deploy_ticks_left = 0
    battle._register(entity)
    return entity


# ------------------------------------------------------------ unit values


def test_a_units_value_is_per_body_not_per_card(world):
    """Skeletons cost one elixir for a handful of bodies.

    Valued per card, a Skeleton would trade as though it cost the same as a
    P.E.K.K.A, and every kite and trade term would read backwards for exactly
    the cheap cards those terms exist to reward.
    """
    _, _, registry = world
    values = unit_elixir_values(registry)
    assert values["Skeleton"] < 1.0, "a Skeleton was valued at a whole card"
    assert values["Knight"] == pytest.approx(3.0)
    assert values["Golem"] > values["Knight"] > values["Skeleton"]


def test_every_deployable_character_has_a_value(world):
    _, _, registry = world
    values = unit_elixir_values(registry)
    missing = [
        character
        for card in registry.standard()
        for character, _ in card.summons()
        if character not in values
    ]
    assert missing == [], missing


# -------------------------------------------------------------------- kite


def test_a_kite_counts_only_when_it_trades_up(world):
    """An Ice Golem on a P.E.K.K.A is a kite. A Knight on a Skeleton is a fight.

    Rewarding the second teaches the agent to trade down, which is the exact
    opposite of the skill the term is meant to capture.
    """
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)

    cheap = _spawn(battle, world, "IceGolemite", Team.BLUE, 9, 12, rarity="Rare")
    expensive = _spawn(battle, world, "Pekka", Team.RED, 9, 12.6, rarity="Epic")
    expensive.target_id = cheap.id
    assert tracker._kiting(battle) > 0, "holding a P.E.K.K.A with an Ice Golem is a kite"

    battle_two = _battle(world)
    tracker_two = RewardTracker(Team.BLUE, registry)
    tracker_two.reset(battle_two)
    blocker = _spawn(battle_two, world, "Knight", Team.BLUE, 9, 12)
    skeleton = _spawn(battle_two, world, "Skeleton", Team.RED, 9, 12.6)
    skeleton.target_id = blocker.id
    assert tracker_two._kiting(battle_two) == 0, "a Knight holding a Skeleton scored as a kite"


def test_an_enemy_attacking_a_tower_is_not_being_kited(world):
    """It is doing exactly what it wanted to do."""
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)

    attacker = _spawn(battle, world, "Pekka", Team.RED, 3.5, 7, rarity="Epic")
    tower = battle._towers[Team.BLUE][0]
    attacker.target_id = tower.id
    assert tracker._kiting(battle) == 0


def test_a_kite_is_worth_more_against_an_expensive_unit(world):
    """Holding a Golem off is worth more than holding a Goblin off."""
    _, _, registry = world

    def held(enemy, rarity):
        battle = _battle(world)
        tracker = RewardTracker(Team.BLUE, registry)
        tracker.reset(battle)
        cheap = _spawn(battle, world, "Skeleton", Team.BLUE, 9, 12)
        foe = _spawn(battle, world, enemy, Team.RED, 9, 12.6, rarity=rarity)
        foe.target_id = cheap.id
        return tracker._kiting(battle)

    assert held("Golem", "Epic") > held("Goblin", "Common") > 0


# ------------------------------------------------------------ elixir trade


def test_the_trade_term_counts_both_sides(world):
    """Destroyed minus lost. A term that only counted kills would reward
    trading a Golem for a Skeleton."""
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)

    mine = _spawn(battle, world, "Knight", Team.BLUE, 9, 12)
    theirs = _spawn(battle, world, "Knight", Team.RED, 9, 20)
    tracker._observe(battle, 1)

    theirs.kill()
    battle._phase_resolve_deaths()
    tracker._observe(battle, 1)
    assert tracker._destroyed > 0 and tracker._lost == 0

    mine.kill()
    battle._phase_resolve_deaths()
    tracker._observe(battle, 1)
    assert tracker._lost > 0


def test_a_death_is_counted_once(world):
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 20)
    tracker._observe(battle, 1)
    victim.kill()
    battle._phase_resolve_deaths()

    tracker._observe(battle, 1)
    once = tracker._destroyed
    for _ in range(5):
        tracker._observe(battle, 1)
    assert tracker._destroyed == once, "a death was counted more than once"


# ------------------------------------------------------------- counterpush


def test_counterpush_is_the_elixir_still_standing(world):
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)
    tracker.score(battle)
    before = tracker.terms["counterpush"]

    _spawn(battle, world, "Golem", Team.BLUE, 9, 12, rarity="Epic")
    tracker.score(battle)
    assert tracker.terms["counterpush"] > before


def test_an_enemy_unit_is_not_your_counterpush(world):
    _, _, registry = world
    battle = _battle(world)
    tracker = RewardTracker(Team.BLUE, registry)
    tracker.reset(battle)
    tracker.score(battle)
    before = tracker.terms["counterpush"]
    _spawn(battle, world, "Golem", Team.RED, 9, 20, rarity="Epic")
    tracker.score(battle)
    assert tracker.terms["counterpush"] == before


# ------------------------------------------------------------ the potential


def test_the_reward_is_the_change_in_the_score(world):
    """Potential-based shaping, so the terms telescope.

    Summed over an episode every shaping term cancels to a constant, which is
    what makes it safe: dense enough to guide the search, and unable to change
    which policy is optimal. Paying a term directly would let the agent farm
    it -- and the obvious farm is elixir, where rewarding units on the board
    without the matching negative when they die is a reward for dumping.
    """
    data, levels, registry = world
    env = CRSimEnv(
        data, levels, registry, DECK, DECK,
        ticks_per_second=20, frame_skip=20, max_ticks=20 * 40,
        reward_weights=RewardWeights(),
    )
    env.reset(seed=4)
    tracker = env._reward
    start = tracker.score(env.battle)

    rng = np.random.default_rng(0)
    total = 0.0
    while True:
        legal = np.argwhere(env.legal_action_mask())
        action = tuple(int(v) for v in legal[rng.integers(len(legal))])
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break

    end = tracker.score(env.battle)
    assert total == pytest.approx(end - start, abs=1e-4), "the reward did not telescope"


def test_every_term_is_finite_and_reported(world):
    data, levels, registry = world
    env = CRSimEnv(
        data, levels, registry, DECK, DECK,
        ticks_per_second=20, frame_skip=20, max_ticks=20 * 40,
        reward_weights=RewardWeights(),
    )
    env.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(20):
        legal = np.argwhere(env.legal_action_mask())
        action = tuple(int(v) for v in legal[rng.integers(len(legal))])
        _, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(reward)
        if terminated or truncated:
            break

    terms = env._reward.terms
    assert set(terms) == {
        "crowns", "tower_damage", "own_tower_hp", "elixir_trade", "counterpush", "kite",
    }
    assert all(np.isfinite(v) for v in terms.values())


def test_the_simple_reward_is_still_available_as_a_control(world):
    """An ablation needs something to ablate against."""
    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, DECK, DECK, ticks_per_second=20, frame_skip=20)
    assert env._reward is None
    env.reset(seed=1)
    _, reward, _, _, _ = env.step((4, 0, 0))
    assert np.isfinite(reward)
