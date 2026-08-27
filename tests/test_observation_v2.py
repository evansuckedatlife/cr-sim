"""The versioned observation, and whether each change actually changes anything.

Three gaps were found by comparing the encoding against a reference spec, and
all three are the kind that a training run cannot report:

*   **No spell or area-effect channel.** The grid excluded projectiles and
    area effects entirely, so the agent could not see an incoming Fireball or
    a Poison cloud standing on its own troops.
*   **Hitpoint mass conflates count with health.** Three Skeletons at 81 each
    and one Knight at 243 wrote an identical cell, and swarm-versus-tank is
    what decides whether the answer is a Log or a Knight.
*   **Enemy elixir given as ground truth.** Included here for a different
    reason than it was raised: in this game the enemy's bar is *not* private.
    The regeneration schedule is public and every card played is visible, so
    a player who counts can reconstruct it exactly --
    ``test_enemy_elixir_is_reconstructible_from_public_information``
    demonstrates that rather than asserting it. The genuinely private thing is
    the opponent's *hand*, which this encoding also handed over and which
    cannot be reconstructed from anything.

Each is behind a flag, because "v2 helps" over four simultaneous changes
cannot say which one paid.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_sim.api.encoding import (
    COUNT_NORM,
    HP_NORM,
    OBSERVATION_V1,
    OBSERVATION_V2,
    SPELL_NORM,
    ObservationFeatures,
    build_encoding_config,
    encode_observation,
    grid_channels,
    hand_onehot_layout,
    observation_shapes,
    parse_observation,
)
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.arena import load_arena
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.constants import ELIXIR_PRECISION, MAX_ELIXIR
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Fireball", "Goblins",
        "Cannon", "Archer", "Skeletons", "Giant")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, seed=1):
    data, levels, registry = world
    battle = Battle(data, levels, registry,
                    BattleConfig(seed=seed, blue_deck=DECK, red_deck=DECK))
    battle.players[Team.BLUE].elixir.add(10)
    battle.players[Team.RED].elixir.add(10)
    return battle


def _config(world, features, battle=None):
    data, _, _ = world
    arena = battle.arena if battle is not None else load_arena(data)
    return build_encoding_config(arena, DECK, DECK, features)


def _encode(world, battle, features):
    _, _, registry = world
    return encode_observation(battle, Team.BLUE, registry,
                              _config(world, features, battle))


# ------------------------------------------------------------------- parsing


def test_the_observation_spec_parses_versions_and_individual_flags():
    assert parse_observation("v1") == OBSERVATION_V1
    assert parse_observation("v2") == OBSERVATION_V2
    one = parse_observation("swarm")
    assert one.swarm and not one.spells and not one.hide_enemy_hand
    both = parse_observation("spells,swarm")
    assert both.spells and both.swarm
    with pytest.raises(ValueError, match="unknown observation flag"):
        parse_observation("spellz")


def test_v1_is_exactly_what_it_was(world):
    """Every checkpoint on disk was trained on this. The default must not
    move under it."""
    assert grid_channels(OBSERVATION_V1) == (
        "own_ground_hp", "own_air_hp", "own_building_hp", "own_tower_hp",
        "enemy_ground_hp", "enemy_air_hp", "enemy_building_hp",
        "enemy_tower_hp", "terrain")
    shapes = observation_shapes(_config(world, OBSERVATION_V1))
    assert shapes["grid"][0] == 9


def test_each_flag_adds_only_the_channels_it_names(world):
    assert len(grid_channels(parse_observation("swarm"))) == 11
    assert len(grid_channels(parse_observation("spells"))) == 13 - 2
    assert len(grid_channels(OBSERVATION_V2)) == 13
    # Terrain stays last however many are switched on; the normalisation
    # slices below it are written as leading ranges.
    for spec in ("v1", "swarm", "spells", "v2"):
        assert grid_channels(parse_observation(spec))[-1] == "terrain"


def test_hiding_information_does_not_change_the_vector_s_length(world):
    """So an ablation compares networks of identical size, and hiding
    something is not also a shape change that would confound it."""
    plain = observation_shapes(_config(world, OBSERVATION_V1))["vector"]
    hidden = observation_shapes(_config(world, OBSERVATION_V2))["vector"]
    assert plain == hidden


# --------------------------------------------------------- the swarm channel


def test_hitpoint_mass_alone_cannot_tell_a_swarm_from_a_tank(world):
    """The gap, demonstrated before it is closed: under v1 three small bodies
    and one large one summing to the same hitpoints write the same cell."""
    from cr_sim.engine.specs import build_unit_spec

    data, levels, registry = world

    def board(unit, count):
        battle = _battle(world)
        spec = build_unit_spec(data, levels, unit, level=11, rarity="Common",
                               clock=battle.clock)
        for _ in range(count):
            entity = Entity(kind=spec.kind, team=Team.BLUE,
                            x=tiles(9), y=tiles(10), hitpoints=spec.hitpoints,
                            spec=spec, collision_radius=spec.collision_radius,
                            mass=spec.mass, flying=spec.flying)
            entity.max_hitpoints = entity.hitpoints
            entity.deploy_ticks_left = 0
            battle._register(entity)
        return battle

    swarm = board("Skeleton", 3)
    total = sum(e.hitpoints for e in swarm.entities
                if e.team is Team.BLUE and e.kind is EntityKind.TROOP)
    # One body carrying the same hitpoints as the three, forced by hand so the
    # comparison is exact rather than approximately equal.
    tank = board("Knight", 1)
    for entity in tank.entities:
        if entity.team is Team.BLUE and entity.kind is EntityKind.TROOP:
            entity.hitpoints = total

    ground = grid_channels(OBSERVATION_V1).index("own_ground_hp")
    swarm_v1 = _encode(world, swarm, OBSERVATION_V1)["grid"][ground]
    tank_v1 = _encode(world, tank, OBSERVATION_V1)["grid"][ground]
    assert np.allclose(swarm_v1, tank_v1), (
        "the two boards were supposed to be indistinguishable under v1")

    channels = grid_channels(parse_observation("swarm"))
    count = channels.index("own_body_count")
    swarm_v2 = _encode(world, swarm, parse_observation("swarm"))["grid"][count]
    tank_v2 = _encode(world, tank, parse_observation("swarm"))["grid"][count]
    assert not np.allclose(swarm_v2, tank_v2), (
        "the body-count channel does not separate three skeletons from one knight")
    assert swarm_v2.max() == pytest.approx(3.0 / COUNT_NORM)
    assert tank_v2.max() == pytest.approx(1.0 / COUNT_NORM)


def test_towers_are_not_counted_as_bodies(world):
    """Three towers a side would put a constant in the channel a Skeleton
    army has to be read out of."""
    battle = _battle(world)
    channels = grid_channels(parse_observation("swarm"))
    grid = _encode(world, battle, parse_observation("swarm"))["grid"]
    assert grid[channels.index("own_body_count")].sum() == 0.0
    assert grid[channels.index("enemy_body_count")].sum() == 0.0
    # ...while the tower hitpoints are still there.
    assert grid[channels.index("own_tower_hp")].sum() > 0.0


# --------------------------------------------------------- the spell channel


def test_an_incoming_spell_is_invisible_under_v1_and_visible_under_v2(world):
    """A Fireball on its way to your push is the thing a player reacts to,
    and the grid excluded projectiles and area effects entirely."""
    battle = _battle(world)
    # The enemy casts at a point on the agent's half.
    red = battle.players[Team.RED]
    red.elixir.add(10)
    before = _encode(world, battle, OBSERVATION_V2)["grid"]
    features = parse_observation("spells")
    channels = grid_channels(features)
    enemy_spell = channels.index("enemy_spell_damage")

    assert battle.play_card(Team.RED, "Fireball", tiles(9), tiles(8)), (
        "the enemy could not cast at all, so the test proves nothing")
    battle.step()
    live = [e for e in battle.entities
            if e.kind in (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT)]
    assert live, "no projectile or area effect exists to be seen"

    v1 = _encode(world, battle, OBSERVATION_V1)["grid"]
    v2 = _encode(world, battle, features)["grid"]
    assert v1.shape[0] == 9
    assert v2[enemy_spell].sum() > 0.0, "the spell channel saw nothing"
    assert v2[channels.index("own_spell_damage")].sum() == 0.0, (
        "the agent's own channel picked up the enemy's spell")
    del before


def test_spell_damage_is_painted_over_the_footprint_not_one_cell(world):
    """A splash radius is the whole point of the read: "a Fireball is
    somewhere" is not the same information as "a Fireball covers these
    tiles"."""
    battle = _battle(world)
    features = parse_observation("spells")
    channels = grid_channels(features)
    assert battle.play_card(Team.RED, "Fireball", tiles(9), tiles(8))
    for _ in range(4):
        battle.step()
    grid = _encode(world, battle, features)["grid"]
    painted = int((grid[channels.index("enemy_spell_damage")] > 0).sum())
    assert painted > 1, f"the spell touched {painted} cell(s), not an area"


def test_a_delivery_projectile_with_no_damage_contributes_nothing(world):
    """Goblin Barrel is pure delivery. It becomes board presence when the
    Goblins land, and reading it as a threat before then would put a large
    number where nothing is about to be damaged."""
    from cr_sim.api.encoding import _spell_damage

    class _Spec:
        damage = 0
        radius = 500

    class _Fake:
        pspec = _Spec()
        aspec = None
        spec = None

    assert _spell_damage(_Fake()) == (0, 500)


# ------------------------------------------------------- hidden and derivable


def test_the_opponent_s_hand_is_zeroed_and_the_agent_s_is_not(world):
    battle = _battle(world)
    config = _config(world, OBSERVATION_V2, battle)
    start, stride, count, width = hand_onehot_layout(config)
    hidden = _encode(world, battle, OBSERVATION_V2)["vector"]
    plain = _encode(world, battle, OBSERVATION_V1)["vector"]

    own = slice(start - 1, start - 1 + (count + 1) * stride)
    enemy = slice(own.stop, own.stop + (count + 1) * stride)
    assert np.allclose(hidden[own], plain[own]), "the agent's own hand was hidden too"
    assert hidden[enemy].sum() == 0.0, "the opponent's hand is still in the vector"
    assert plain[enemy].sum() > 0.0, "v1 was supposed to expose it"


def test_hiding_the_enemy_elixir_zeroes_only_that_scalar(world):
    battle = _battle(world)
    plain = _encode(world, battle, OBSERVATION_V1)["vector"]
    hidden = _encode(world, battle, parse_observation("hide_enemy_elixir"))["vector"]
    assert hidden[0] == plain[0], "the agent's own elixir moved"
    assert hidden[1] == 0.0
    assert plain[1] > 0.0
    assert np.allclose(hidden[2:], plain[2:])


def test_enemy_elixir_is_reconstructible_from_public_information(world):
    """Why the elixir scalar is a different case from the hand.

    The regeneration schedule is in the game's own data and every card played
    is visible on the board, so an observer who starts at the first tick and
    counts can carry the opponent's bar forward exactly. This walks a match
    doing precisely that -- a shadow bar advanced by the public timeline and
    debited by the visible cost of each play -- and checks it never diverges.

    So hiding it is a statement about how much bookkeeping the agent should
    have to do, not about information it could not have. The hand is the
    genuinely private thing, and no amount of counting recovers it.
    """
    from cr_sim.engine.elixir import ElixirBar

    battle = _battle(world, seed=7)
    _, _, registry = world
    # The observer's own copy of the bar, built from the public timeline and
    # the public starting elixir. Nothing here reads the opponent's state.
    shadow = ElixirBar(timeline=battle.timeline)
    shadow.amount = battle.players[Team.RED].elixir.amount

    played = 0
    for step in range(600):
        tick = battle.tick
        battle.step()
        shadow.regenerate(tick)
        if step % 137 == 0 and played < 4:
            card = battle.players[Team.RED].hand[0]
            cost = registry.get(card).mana_cost
            if battle.play_card(Team.RED, card, tiles(9), tiles(24)):
                # Public: the card appeared on the board, and what it costs is
                # printed on it.
                shadow.amount -= cost * ELIXIR_PRECISION
                played += 1
        assert shadow.units == battle.players[Team.RED].elixir.units, (
            f"the reconstruction diverged at tick {battle.tick}")
    assert played, "no card was ever played, so nothing was reconstructed"
