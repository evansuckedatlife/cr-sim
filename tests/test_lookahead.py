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
