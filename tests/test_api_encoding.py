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

import hashlib

import numpy as np
import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.arena import load_arena
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec

from cr_sim.api.encoding import (
    DPS_NORM,
    GRID_CHANNELS,
    N_GRID_CHANNELS,
    NOOP_SLOT,
    NUM_CARD_SLOTS,
    PLACEMENT_TILE_SPAN,
    REACH_NORM,
    _placement_grid,
    action_grid_shape,
    build_encoding_config,
    cell_to_world,
    decode_action,
    encode_observation,
    grid_channels,
    legal_action_mask,
    observation_shapes,
    parse_observation,
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


# ------------------------------------------------------------ threat channels
#
# What a cell can *do*, as opposed to how much of it there is. Hitpoint mass
# alone puts a Musketeer and a Knight in a cell as roughly the same number,
# and the difference between them -- 6.0 tiles of reach against 1.2, so she
# kills him without ever being touched -- is the whole fight. These check that
# the set is four channels wide, that it fills with the values the units
# actually have, and that switching it off leaves the encoding every
# checkpoint on disk was trained on byte for byte where it was.

THREAT = "threat"


def _threat_channels():
    return grid_channels(parse_observation(THREAT))


def _encode(world, battle, spec=None):
    """``battle`` through BLUE's eyes, under ``spec``'s feature set (v1 if None)."""
    data, _levels, registry = world
    arena = load_arena(data)
    config = (build_encoding_config(arena, DECK, DECK, parse_observation(spec)) if spec
              else build_encoding_config(arena, DECK, DECK))
    return encode_observation(battle, Team.BLUE, registry, config)


def _stacked(world, unit, count, *, x=9, y=12):
    """A bare board -- towers stripped -- with ``count`` of ``unit`` on one cell."""
    battle = _bare(_battle(world, seed=7))
    for _ in range(count):
        _spawn(battle, world, unit, Team.BLUE, x, y)
    return battle


def _dps_reach(world, battle):
    channels = _threat_channels()
    grid = _encode(world, battle, THREAT)["grid"]
    return grid[channels.index("own_dps")], grid[channels.index("own_reach")]


def test_the_threat_set_adds_four_channels_and_moves_none_of_the_others(world):
    """One entry in the registry, four channels, terrain still last -- and the
    eight hitpoint channels bit-identical, so the set is additive rather than a
    rearrangement that would quietly repoint every existing conv filter."""
    channels = _threat_channels()
    assert channels == GRID_CHANNELS[:-1] + (
        "own_dps", "enemy_dps", "own_reach", "enemy_reach", "terrain")
    assert len(channels) == N_GRID_CHANNELS + 4 == 13

    battle = _battle(world, seed=3, hand=HAND)
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    for _ in range(60):
        battle.step()
    plain = _encode(world, battle)["grid"]
    threat = _encode(world, battle, THREAT)["grid"]
    assert np.array_equal(threat[:N_GRID_CHANNELS - 1], plain[:-1]), (
        "enabling threat moved the hitpoint channels")
    assert np.array_equal(threat[-1], plain[-1]), "terrain did not stay last"


def test_leaving_the_threat_set_off_encodes_exactly_what_it_did_before(world):
    """The default observation is what every checkpoint in ``runs/`` was
    trained on, so this pins the bytes and not just the shape: a channel that
    displaced an existing one would not fail on a shape mismatch, it would
    just make an old policy read the wrong plane and play badly.

    The digest was taken from this same battle before the threat set existed.
    If it moves, the default encoding moved with it.
    """
    battle = _battle(world, seed=3, hand=HAND)
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    for _ in range(120):
        battle.step()
    obs = _encode(world, battle)
    assert obs["grid"].shape == (9, 32, 18)
    digest = hashlib.sha256()
    digest.update(obs["grid"].tobytes())
    digest.update(obs["vector"].tobytes())
    assert digest.hexdigest() == (
        "771c41111b29335f88c4fddd356850c3601c0c03fd1233378c2aa0653829b915")


def test_damage_sums_over_a_stack_while_reach_takes_the_cell_maximum(world):
    """The one decision in this set that is not obvious. Two Musketeers in a
    cell put out twice the damage but do not shoot a tile further, so the two
    halves accumulate differently: summing reach as well would read a pair of
    Musketeers as a 12-tile siege weapon.
    """
    data, levels, _registry = world
    one, two = _stacked(world, "Musketeer", 1), _stacked(world, "Musketeer", 2)
    spec = build_unit_spec(data, levels, "Musketeer", level=11, rarity="Common",
                           clock=one.clock)
    # Sanity on the fixture itself: a Musketeer really is the long-reach,
    # moderate-damage unit the rest of this reasons about.
    assert spec.damage_per_second == pytest.approx(217.0, abs=0.1)
    assert to_tiles(spec.attack_range) == pytest.approx(6.0)

    one_dps, one_reach = _dps_reach(world, one)
    two_dps, two_reach = _dps_reach(world, two)
    assert one_dps.max() == pytest.approx(spec.damage_per_second / DPS_NORM)
    assert two_dps.max() == pytest.approx(2 * spec.damage_per_second / DPS_NORM)
    assert one_reach.max() == pytest.approx(6.0 / REACH_NORM)
    assert two_reach.max() == pytest.approx(6.0 / REACH_NORM), (
        "two Musketeers were read as reaching further than one")
    # A whole card's worth of damage still reads below saturation, which is
    # what DPS_NORM was set above the build's hardest hitter for.
    assert 0.0 < two_dps.max() < 1.0


def test_a_musketeer_and_a_knight_are_one_number_under_v1_and_two_under_threat(world):
    """The gap the set closes, demonstrated the way the swarm channel's was:
    the same hitpoints in the same cell, indistinguishable until the threat
    channels separate them."""
    musketeer = _stacked(world, "Musketeer", 1)
    knight = _stacked(world, "Knight", 1)
    # Forced equal so the v1 comparison is exact rather than approximate. The
    # two cards' real hitpoints differ; the point is that mass is all v1 has
    # to tell them apart with.
    hitpoints = next(e.hitpoints for e in musketeer.entities)
    for entity in knight.entities:
        entity.hitpoints = hitpoints

    ground = GRID_CHANNELS.index("own_ground_hp")
    assert np.array_equal(_encode(world, musketeer)["grid"][ground],
                          _encode(world, knight)["grid"][ground]), (
        "the two boards were supposed to be indistinguishable under v1")

    _, musketeer_reach = _dps_reach(world, musketeer)
    _, knight_reach = _dps_reach(world, knight)
    gap = (musketeer_reach.max() - knight_reach.max()) * REACH_NORM
    assert gap == pytest.approx(4.8, abs=0.05), (
        f"the reach channel put {gap:.2f} tiles between a Musketeer and a "
        f"Knight; the measured difference is 4.8")


def test_the_towers_are_the_boards_baseline_threat(world):
    """A Princess Tower is the reason a defence that would lose on its own
    holds, so it belongs in these channels -- unlike the body-count channel,
    which deliberately leaves towers out. On an otherwise empty board the
    threat channels are exactly the three towers a side and nothing else.
    """
    battle = _battle(world, seed=1)
    channels = _threat_channels()
    grid = _encode(world, battle, THREAT)["grid"]
    for side in ("own", "enemy"):
        dps = grid[channels.index(f"{side}_dps")]
        reach = grid[channels.index(f"{side}_reach")]
        # King 92.0 dps at 7.0 tiles, two Princesses 115.0 at 7.5.
        assert dps.sum() == pytest.approx((92.0 + 115.0 + 115.0) / DPS_NORM, rel=1e-4)
        assert reach.max() == pytest.approx(7.5 / REACH_NORM)
        assert int((dps > 0).sum()) == 3, "a tower is missing from the threat channels"


def test_threat_channels_populate_and_stay_normalised_in_a_real_battle(world):
    """End to end rather than on a hand-built board: play a card, run the
    engine, and check the channels carry the unit that was played on top of
    the tower baseline -- and that nothing leaves [0, 1]."""
    channels = _threat_channels()
    data, levels, _registry = world

    quiet = _battle(world, seed=3, hand=HAND)
    baseline = _encode(world, quiet, THREAT)["grid"][channels.index("own_dps")].sum()

    battle = _battle(world, seed=3, hand=HAND)
    assert battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12)), (
        "the Knight was never played, so the test proves nothing")
    for _ in range(120):
        battle.step()
    grid = _encode(world, battle, THREAT)["grid"]
    knight = build_unit_spec(data, levels, "Knight", level=11, rarity="Common",
                             clock=battle.clock)

    added = grid[channels.index("own_dps")].sum() - baseline
    assert added == pytest.approx(knight.damage_per_second / DPS_NORM, rel=1e-3), (
        "the played Knight did not show up in the damage channel")
    assert grid[channels.index("enemy_dps")].sum() > 0.0, "the enemy towers vanished"
    assert grid[channels.index("enemy_reach")].max() > 0.0
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)
    assert np.all(np.isfinite(grid))


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


def _red_left_princess(battle):
    """RED's left-lane Princess Tower -- the one BLUE's expanded-zone tests
    below take down. Read from ``battle._towers`` (not ``battle.entities``),
    the same index ``Battle._king`` and ``Battle.fallen_enemy_towers`` use,
    since a destroyed tower is retired out of the live entity list.
    """
    return min(
        (t for t in battle._towers[Team.RED] if "King" not in t.spec.name),
        key=lambda t: t.x,
    )


def test_legal_mask_grows_once_a_princess_tower_falls(world):
    """An RL agent only ever samples from this mask -- if destroying a tower
    does not make it grow, the agent can never learn to use the expanded
    zone, however good the resulting push would be.

    Both calls run against the very same ``Battle`` (and therefore the very
    same ``Arena`` object): this is also the sharpest check that
    ``_placement_grid``'s cache is not serving the pre-kill grid back for the
    second call, since nothing about object identity changes between them --
    only which towers are dead.
    """
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    knight_slot = HAND.index("Knight")

    battle = _battle(world, seed=5, hand=HAND)
    before = legal_action_mask(battle, Team.BLUE, registry, config)

    _red_left_princess(battle).kill()
    battle.step()
    after = legal_action_mask(battle, Team.BLUE, registry, config)

    assert after[knight_slot].sum() > before[knight_slot].sum(), (
        "the mask did not grow after a Princess Tower fell -- either the "
        "expanded zone is not implemented, or the placement-grid cache is "
        "serving a stale, pre-kill grid"
    )


def test_legal_mask_reflects_the_expanded_zone_and_agrees_with_play_card(world):
    """The mask growing (previous test) is only half the story -- the new
    cells it marks legal have to actually BE legal, checked the same way
    ``test_legal_mask_matches_actual_play_card_legality`` checks the rest of
    the mask: against ``Battle.play_card`` itself.
    """
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    knight_slot = HAND.index("Knight")

    before = legal_action_mask(_battle(world, seed=5, hand=HAND), Team.BLUE, registry, config)

    mask_battle = _battle(world, seed=5, hand=HAND)
    _red_left_princess(mask_battle).kill()
    mask_battle.step()
    after = legal_action_mask(mask_battle, Team.BLUE, registry, config)

    newly_legal = np.argwhere(after[knight_slot] & ~before[knight_slot])
    assert newly_legal.size > 0, "destroying a Princess Tower added no new legal cells"

    gx, gy = (int(v) for v in newly_legal[0])
    x, y = cell_to_world(gx, gy, Team.BLUE, arena, span=PLACEMENT_TILE_SPAN)

    play_battle = _battle(world, seed=5, hand=HAND)
    _red_left_princess(play_battle).kill()
    play_battle.step()
    assert play_battle.play_card(Team.BLUE, "Knight", x, y), (
        f"mask marked cell ({gx},{gy}) newly legal after a tower fell, but "
        f"play_card refused it at the same point"
    )


def test_the_placement_grid_cache_does_not_serve_a_stale_zone(world):
    """Direct check on ``_placement_grid`` itself -- the private cache the bug
    report specifically called out. It is keyed on ``(arena, team, anywhere,
    on_water, width, height, fallen_enemy_towers)``; two calls that differ
    only in the last of those must not come back equal, and two calls that
    agree on all of them (including an empty ``fallen_enemy_towers``) must
    still hit the same cached grid rather than colliding with each other.
    """
    data, levels, registry = world
    arena = load_arena(data)
    config = build_encoding_config(arena, DECK, DECK)
    width, height = config.action_width, config.action_height

    red_left = min(arena.princess_towers(Team.RED), key=lambda t: t.x)

    grid_before = _placement_grid(arena, Team.BLUE, False, False, width, height)
    grid_after = _placement_grid(
        arena, Team.BLUE, False, False, width, height, frozenset({red_left})
    )
    assert not np.array_equal(grid_before, grid_after), (
        "the same arena/team/flags but a different fallen-tower set produced "
        "an identical grid -- the cache key is not tracking tower state"
    )
    # Repeating the original, tower-less call afterwards must still return the
    # unexpanded grid: the two keys must not collide with each other either.
    assert np.array_equal(
        _placement_grid(arena, Team.BLUE, False, False, width, height), grid_before
    )


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
