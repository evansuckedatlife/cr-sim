"""M8 gate: observation and action encoding.

The risk this module exists to catch is not "does the code run" -- it is "does
the legality mask agree with the engine it is describing." An RL agent only
ever sees the mask, never ``Arena.can_deploy`` directly, so a mask that is
wrong in either direction is invisible to training and silently corrupts
every episode: too permissive wastes samples on actions the engine refuses,
too restrictive hides real options from the policy. That is what
``test_legal_mask_matches_actual_play_card_legality`` checks end to end,
against ``Battle.play_card`` itself rather than against a second
reimplementation of the placement rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.arena import load_arena
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles
from cr_sim.engine.specs import build_unit_spec

from cr_sim.api.encoding import (
    GRID_CHANNELS,
    N_GRID_CHANNELS,
    NOOP_SLOT,
    NUM_CARD_SLOTS,
    PLACEMENT_TILE_SPAN,
    action_grid_shape,
    build_encoding_config,
    cell_to_world,
    decode_action,
    encode_observation,
    legal_action_mask,
    observation_shapes,
)

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Fireball", "Goblins", "Cannon", "Archer", "Skeletons", "Giant")
#: A hand fixed by construction rather than left to the seed's shuffle, so the
#: mask test exercises one troop, one building and one spell -- the three
#: placement-rule shapes Arena.can_deploy actually distinguishes -- on every run.
HAND = ("Knight", "Fireball", "Giant", "Cannon")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, *, seed=1, hand=None):
    data, levels, registry = world
    battle = Battle(data, levels, registry, BattleConfig(seed=seed, blue_deck=DECK, red_deck=DECK))
    if hand is not None:
        rest = [c for c in DECK if c not in hand]
        battle.players[Team.BLUE].cycle = list(hand) + rest
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _spawn(battle, world, unit, team, x, y):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, unit, level=11, rarity="Common", clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y),
        hitpoints=spec.hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = entity.hitpoints
    entity.deploy_ticks_left = 0
    battle._register(entity)
    return entity


def _bare(battle):
    """Strip towers so a placed unit is the only thing on the board."""
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    return battle


# --------------------------------------------------------------- observation


def test_observation_shape_and_dtype_are_stable_across_resets_and_steps(world):
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    shapes = observation_shapes(config)

    battle_a = _battle(world, seed=1)
    obs_a = encode_observation(battle_a, Team.BLUE, registry, config)
    assert obs_a["grid"].shape == shapes["grid"]
    assert obs_a["vector"].shape == shapes["vector"]
    assert obs_a["grid"].dtype == np.float32
    assert obs_a["vector"].dtype == np.float32

    # A different seed is the same thing a reset() does: a brand new Battle.
    battle_b = _battle(world, seed=2)
    obs_b = encode_observation(battle_b, Team.BLUE, registry, config)
    assert obs_b["grid"].shape == obs_a["grid"].shape
    assert obs_b["vector"].shape == obs_a["vector"].shape

    for _ in range(30):
        battle_a.step()
    obs_a2 = encode_observation(battle_a, Team.BLUE, registry, config)
    assert obs_a2["grid"].shape == obs_a["grid"].shape
    assert obs_a2["vector"].shape == obs_a["vector"].shape
    assert obs_a2["grid"].dtype == np.float32
    assert obs_a2["vector"].dtype == np.float32


def test_observation_values_stay_normalised(world):
    """Everything is documented as living in [0, 1]; this is what enforces it."""
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    battle = _battle(world, seed=3, hand=HAND)
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    for _ in range(120):
        battle.step()
    obs = encode_observation(battle, Team.BLUE, registry, config)
    assert np.all(obs["grid"] >= 0.0) and np.all(obs["grid"] <= 1.0)
    assert np.all(obs["vector"] >= 0.0) and np.all(obs["vector"] <= 1.0)
    assert np.all(np.isfinite(obs["grid"]))
    assert np.all(np.isfinite(obs["vector"]))


def test_grid_places_entities_at_the_mirrored_own_perspective_cell(world):
    """A troop near a team's own baseline must land at the same grid cell in
    that team's own-perspective view as its mirror image does for the other
    team -- otherwise a shared-weights self-play policy sees two different
    boards for what is tactically the same position.
    """
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    own_ground = GRID_CHANNELS.index("own_ground_hp")

    blue_battle = _bare(_battle(world, seed=11))
    _spawn(blue_battle, world, "Knight", Team.BLUE, 9, 2)
    obs_blue = encode_observation(blue_battle, Team.BLUE, registry, config)

    red_battle = _bare(_battle(world, seed=11))
    _spawn(red_battle, world, "Knight", Team.RED, 9, 30)  # the y-mirror of (9, 2)
    obs_red = encode_observation(red_battle, Team.RED, registry, config)

    blue_cells = np.argwhere(obs_blue["grid"][own_ground] > 0).tolist()
    red_cells = np.argwhere(obs_red["grid"][own_ground] > 0).tolist()
    assert blue_cells == red_cells == [[2, 9]]


# -------------------------------------------------------------------- action


def test_action_grid_matches_the_documented_resolution(world):
    """A regression guard on the resolution the module docstring's whole
    argument (720 actions, not 2880) depends on."""
    data, _levels, _registry = world
    arena = load_arena(data)
    assert action_grid_shape(arena) == (9, 16)
    assert NUM_CARD_SLOTS * 9 * 16 == 720


def test_cell_to_world_always_lands_in_bounds(world):
    data, _levels, _registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    for team in (Team.BLUE, Team.RED):
        for gx in (0, config.action_width - 1):
            for gy in (0, config.action_height - 1):
                x, y = cell_to_world(gx, gy, team, arena, span=PLACEMENT_TILE_SPAN)
                assert 0 <= x < arena.width
                assert 0 <= y < arena.height


def test_decode_action_is_none_for_the_noop_slot(world):
    data, _levels, _registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    assert decode_action((NOOP_SLOT, 0, 0), Team.BLUE, arena, config) is None
    assert decode_action((0, 0, 0), Team.BLUE, arena, config) is not None


def test_decode_action_rejects_an_out_of_range_slot(world):
    data, _levels, _registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    with pytest.raises(ValueError):
        decode_action((NUM_CARD_SLOTS, 0, 0), Team.BLUE, arena, config)


# ---------------------------------------------------------------- legality


def test_legal_mask_matches_actual_play_card_legality(world):
    """The most important test in this file. For a sample of cells across
    one troop, one building and one spell, the mask's verdict must match
    what ``Battle.play_card`` itself does at that exact point.
    """
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    width, height = config.action_width, config.action_height

    mask_battle = _battle(world, seed=5, hand=HAND)
    mask = legal_action_mask(mask_battle, Team.BLUE, registry, config)
    # One cell, not the whole slot: passing has no position, so the other
    # 143 cells would be duplicates competing for probability mass.
    assert mask[NOOP_SLOT].sum() == 1, "passing should be exactly one action"
    assert mask[NOOP_SLOT, 0, 0], "passing must always be legal"

    sample = [(gx, gy) for gx in range(0, width, 3) for gy in range(0, height, 4)]
    legal_checked = illegal_checked = 0
    for slot, card_name in enumerate(HAND):
        for gx, gy in sample:
            fresh = _battle(world, seed=5, hand=HAND)
            assert fresh.players[Team.BLUE].hand[slot] == card_name
            x, y = cell_to_world(gx, gy, Team.BLUE, arena, span=PLACEMENT_TILE_SPAN)
            played = fresh.play_card(Team.BLUE, card_name, x, y)
            is_legal = bool(mask[slot, gx, gy])
            if is_legal:
                assert played, (
                    f"mask marked {card_name} legal at cell ({gx},{gy}) "
                    f"but play_card refused it"
                )
                legal_checked += 1
            else:
                assert not played, (
                    f"mask marked {card_name} illegal at cell ({gx},{gy}) "
                    f"but play_card accepted it"
                )
                illegal_checked += 1

    # A sample that never saw a legal or never saw an illegal cell would let
    # either half of the mask be silently wrong without either assert above
    # ever firing -- both counters have to be nonzero for the test to mean
    # anything.
    assert legal_checked > 0, "sample contained no legal actions to check"
    assert illegal_checked > 0, "sample contained no illegal actions to check"


def test_legal_mask_respects_affordability(world):
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    battle = _battle(world, seed=9, hand=HAND)
    battle.players[Team.BLUE].elixir.amount = 0

    mask = legal_action_mask(battle, Team.BLUE, registry, config)
    assert not mask[:NOOP_SLOT].any(), "no elixir: nothing should be affordable"
    assert mask[NOOP_SLOT, 0, 0], "passing never costs elixir"


def test_legal_mask_shape_matches_action_grid(world):
    data, _levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    battle = _battle(world, seed=5, hand=HAND)
    mask = legal_action_mask(battle, Team.BLUE, registry, config)
    # (slot, x, y) -- the same order an action tuple is written, so a mask
    # index can be used as an action without transposing.
    assert mask.shape == (NUM_CARD_SLOTS, config.action_width, config.action_height)
    assert mask.dtype == np.bool_
