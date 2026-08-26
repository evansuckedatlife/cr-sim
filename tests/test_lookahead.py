"""Branching a battle, and reading what the board is already worth.

The engine is deterministic, so from any position there is exactly one answer
to "what happens if nobody plays another card". That makes an *exact* board
evaluation available for the cost of a clone plus some ticks -- no training, no
calibration, nothing to mis-weight. It exists because the learned critic was
the weak link: measured on its own training distribution it explained six per
cent of the variance in returns, which leaves PPO's advantages mostly noise.

Everything here rests on the clone being a complete copy. A clone that shared
one mutable structure by accident would let a throwaway branch corrupt the real
match, and the symptom would look like nondeterminism rather than like aliasing
-- so the first two tests are worth more than the rest put together.
"""

from __future__ import annotations

import pytest

from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import EntityKind, Team
from cr_sim.engine.fixed import tiles
from cr_sim.engine.lookahead import committed_value, project
from cr_sim.replay import state_hash

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    from cr_sim.data.cards import build_card_registry
    from cr_sim.data.leveling import build_level_table
    from cr_sim.data.source import LogicData

    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, ticks=400, seed=1):
    data, levels, registry = world
    battle = Battle(data, levels, registry, BattleConfig(
        seed=seed, ticks_per_second=20, blue_deck=DECK, red_deck=DECK))
    for _ in range(ticks):
        battle.step()
    return battle


def _hash(battle):
    return state_hash(battle.tick, battle.entities)


def _with_knight(world, x=1, y=2):
    battle = _battle(world)
    assert battle.play_card(Team.BLUE, "Knight", tiles(x), tiles(y)), "placement rejected"
    for _ in range(30):
        battle.step()
    return battle


# ------------------------------------------------------------------- cloning


def test_a_branch_replays_its_origin_tick_for_tick(world):
    """The completeness test: any state the clone shared or missed shows here.

    The two are advanced separately rather than in lockstep, because entity
    ids come from a module-level counter and interleaving two battles would
    hand them alternating ids -- a divergence in the bookkeeping rather than
    in the simulation. Winding the counter back between the two runs is what
    makes them comparable, and is the same thing :func:`project` does so a
    projection cannot perturb the battle it was asked about.
    """
    from cr_sim.engine.entity import entity_id_cursor, restore_entity_ids

    battle = _with_knight(world)
    branch = battle.clone()

    cursor = entity_id_cursor()
    original = []
    for _ in range(300):
        battle.step()
        original.append(_hash(battle))

    restore_entity_ids(cursor)
    for step, expected in enumerate(original):
        branch.step()
        assert _hash(branch) == expected, f"diverged {step} ticks after cloning"


def test_projecting_does_not_consume_entity_ids(world):
    """A projection must be invisible to the battle it is asked about.

    It is not enough that the branch is discarded: the id counter is global,
    so a branch that spawned anything would leave the live battle handing out
    different ids than it would have otherwise, and the match would play out
    differently for having been evaluated. That is a determinism bug, and it
    would surface as an unreproducible replay rather than as anything that
    points back here.
    """
    from cr_sim.engine.entity import entity_id_cursor

    battle = _with_knight(world)
    before = entity_id_cursor()
    projection = project(battle)
    assert projection.ticks > 0, "nothing was simulated, so nothing was proven"
    assert entity_id_cursor() == before


def test_playing_a_branch_forward_leaves_its_origin_untouched(world):
    battle = _with_knight(world)
    before, tick, count = _hash(battle), battle.tick, len(battle.entities)
    graves, damage = len(battle.graveyard), len(battle.damage_log)

    branch = battle.clone()
    for _ in range(600):
        branch.step()

    assert _hash(battle) == before
    assert (battle.tick, len(battle.entities)) == (tick, count)
    # The append-only histories are shared by element and copied by container;
    # a branch appending to its own must not extend the original's.
    assert (len(battle.graveyard), len(battle.damage_log)) == (graves, damage)


def test_a_branch_does_not_inherit_frame_recording(world):
    """Frames are for the viewer, and nobody watches a discarded branch."""
    assert _with_knight(world).clone().frames == []


def test_two_branches_from_one_position_agree(world):
    """Determinism survives branching: same position, same future."""
    battle = _with_knight(world)
    first, second = battle.clone(), battle.clone()
    for _ in range(200):
        first.step()
        second.step()
    assert _hash(first) == _hash(second)


# ---------------------------------------------------------------- projecting


def test_a_quiet_board_projects_to_itself_without_simulating(world):
    """Towers alone cannot hurt each other, so there is nothing to compute.

    Worth its own path rather than falling out of the simulation: most
    decisions in a match are taken on an empty board, and the early exit is
    two orders of magnitude cheaper than the clone it avoids.
    """
    projection = project(_battle(world, ticks=100))
    assert projection.ticks == 0
    assert projection.blue_tower_fraction == 1.0
    assert projection.red_tower_fraction == 1.0


def test_a_committed_push_projects_damage_to_the_defending_towers(world):
    """One Knight, nothing answering it, and the projection says so."""
    projection = project(_with_knight(world))
    assert projection.red_tower_fraction < 1.0, "an unanswered Knight did no damage"
    assert projection.blue_tower_fraction == 1.0, "nothing was attacking blue"
    assert projection.decided, "running to the end should reach a result"


def test_a_horizon_stops_early_and_says_so(world):
    projection = project(_with_knight(world), horizon_ticks=40)
    assert projection.ticks == 40
    assert not projection.decided


def test_the_value_of_a_board_is_opposite_for_the_two_sides(world):
    """One side's committed advantage is the other's committed deficit."""
    battle = _with_knight(world)
    blue = committed_value(battle, Team.BLUE)
    red = committed_value(battle, Team.RED)
    assert blue == pytest.approx(-red)
    assert blue > 0, "blue committed a Knight against an empty board"


def test_projecting_does_not_disturb_the_battle_being_projected(world):
    battle = _with_knight(world)
    before = _hash(battle)
    project(battle)
    project(battle, horizon_ticks=100)
    committed_value(battle, Team.BLUE)
    assert _hash(battle) == before


def test_a_longer_horizon_sees_at_least_as_much_damage(world):
    """The Knight is walking towards the towers, so waiting cannot un-hit them."""
    battle = _with_knight(world)
    near = project(battle, horizon_ticks=100).red_tower_fraction
    far = project(battle, horizon_ticks=600).red_tower_fraction
    assert far <= near


# ------------------------------------------------------- as a reward potential


def _env(world, weights, seed=0):
    from cr_sim.api.env import CRSimEnv

    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, DECK, DECK,
                   ticks_per_second=20, frame_skip=30, max_ticks=20 * 60,
                   reward_shaping_weight=0.01, reward_weights=weights)
    env.reset(seed=seed)
    return env


def _play_out(env, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    slots, width, height = env.action_space.nvec
    rewards = []
    while True:
        mask = env.legal_action_mask().reshape(-1)
        index = int(rng.choice(np.flatnonzero(mask))) if mask.any() else 0
        slot, remainder = divmod(index, width * height)
        gx, gy = divmod(remainder, height)
        _, reward, terminated, truncated, _ = env.step(
            (min(slot, slots - 1), gx, gy))
        rewards.append(reward)
        if terminated or truncated:
            return rewards


def test_the_episode_reward_telescopes_to_the_change_in_potential(world):
    """The defining property of potential-based shaping, and the reason this
    is safe to use: the rewards sum to the difference in potential between the
    first and last state, so no policy can farm the shaping for return that
    the outcome did not earn.
    """
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    weights = ProjectionWeights(horizon_seconds=2.0)
    env = _env(world, weights, seed=4)
    tracker = ProjectedReward(Team.BLUE, weights)
    start = tracker.score(env.battle)

    total = sum(_play_out(env, seed=4))
    end = tracker.score(env.battle)
    assert total == pytest.approx(end - start, abs=1e-6)


def test_the_potential_is_opposite_for_the_two_sides(world):
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    weights = ProjectionWeights(horizon_seconds=2.0)
    battle = _with_knight(world)
    blue = ProjectedReward(Team.BLUE, weights).score(battle)
    red = ProjectedReward(Team.RED, weights).score(battle)
    assert blue == pytest.approx(-red)


def test_spending_elixir_costs_potential_immediately(world):
    """What makes a play have to justify itself.

    A card leaves the hand before it has done anything, so the elixir term
    drops the moment it is spent. Without that, the potential could only ever
    go up when playing a card and 'spend everything immediately' would be free.
    """
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    weights = ProjectionWeights(horizon_seconds=2.0)
    tracker = ProjectedReward(Team.BLUE, weights)
    battle = _battle(world)
    before = tracker.score(battle)
    assert battle.play_card(Team.BLUE, "Knight", tiles(1), tiles(2))
    after = tracker.score(battle)
    assert after < before, "a card that has not deployed yet was free"


def test_a_quiet_board_pays_nothing(world):
    """Neither side committing anything is not progress in either direction."""
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    tracker = ProjectedReward(Team.BLUE, ProjectionWeights(horizon_seconds=2.0))
    battle = _battle(world, ticks=100)
    tracker.reset(battle)
    for _ in range(60):
        battle.step()
    assert tracker.step(battle, 60) == pytest.approx(0.0, abs=1e-9)


def test_the_terms_are_reported_separately(world):
    """One number cannot say whether the agent is attacking or hoarding."""
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    tracker = ProjectedReward(Team.BLUE, ProjectionWeights(horizon_seconds=2.0))
    tracker.score(_with_knight(world))
    assert set(tracker.terms) == {"crowns", "towers", "elixir", "projected_ticks"}


# ------------------------------------------- skipping the intermediate scores


def test_skipping_intermediate_scores_pays_exactly_the_same_reward(world):
    """The optimisation that makes a projected-reward run affordable.

    Most states in a match are forced -- one legal action -- and the
    environment runs through them without consulting the policy. It used to
    score the reward at every one of them, which meant a board projection
    each, and those were about two thirds of the run's compute.

    A potential telescopes: the intermediate terms cancel, so scoring only the
    endpoints is arithmetically identical rather than an approximation. This
    proves that on a real episode instead of asserting it, because "identical"
    is exactly the kind of claim that is true right up until someone adds a
    path-dependent term to the potential.
    """
    from cr_sim.api.env import CRSimEnv
    from cr_sim.api.reward import ProjectedReward, ProjectionWeights

    data, levels, registry = world
    weights = ProjectionWeights(horizon_seconds=2.0)

    def total(telescoping: bool) -> float:
        env = CRSimEnv(data, levels, registry, DECK, DECK,
                       ticks_per_second=20, frame_skip=30, max_ticks=20 * 60,
                       reward_shaping_weight=0.01, reward_weights=weights)
        # Same class either way; only the shortcut is toggled, so any
        # difference is the shortcut's fault and nothing else's.
        env._reward = ProjectedReward(env.team, weights)
        # The flag is read off the class, so that is where it is set.
        ProjectedReward.telescopes = telescoping
        env.reset(seed=11)
        return sum(_play_out(env, seed=11))

    try:
        with_skip = total(True)
        without = total(False)
    finally:
        ProjectedReward.telescopes = True

    assert with_skip == pytest.approx(without, abs=1e-9), (
        "skipping intermediate scores changed the reward, so the potential "
        "is not telescoping the way the shortcut assumes")
