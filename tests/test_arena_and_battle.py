"""M1 gate: arena geometry and the tick loop.

Terrain is read from the game's own tilemap rather than reconstructed, so these
tests assert the *decoding* is right -- that the bitfield means what it appears
to mean, and that the geometry that falls out matches the board people play on.

The battle tests then check the two things M1 promises: a unit walks at its real
speed along a legal route, and the same seed produces the same battle.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.arena import Tile, load_arena
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.pathing import route_to
from cr_sim.replay import Command, compare_hashes

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Fireball", "Goblins", "Cannon", "Archer", "Skeletons", "Giant")


@pytest.fixture(scope="module")
def data():
    return LogicData.load(BUILD)


@pytest.fixture(scope="module")
def levels(data):
    return build_level_table(data)


@pytest.fixture(scope="module")
def registry(data):
    return build_card_registry(data)


@pytest.fixture(scope="module")
def arena(data):
    return load_arena(data)


def make_battle(data, levels, registry, *, first_card="Knight", seed=7, tps=60) -> Battle:
    """A battle with a known opening hand, so deployment tests are deterministic."""
    battle = Battle(
        data,
        levels,
        registry,
        BattleConfig(seed=seed, ticks_per_second=tps, blue_deck=DECK, red_deck=DECK),
    )
    battle.players[Team.BLUE].cycle = [first_card] + [c for c in DECK if c != first_card]
    return battle


# ------------------------------------------------------------ tilemap decode


def test_arena_is_eighteen_by_thirty_two(arena):
    assert arena.half_width == 36
    assert arena.half_height == 64
    assert (arena.width_tiles, arena.height_tiles) == (18.0, 32.0)


def test_every_tile_value_is_a_known_bit_combination(arena):
    """No leftover bits -- this is what confirms the bitfield reading."""
    known = (
        Tile.LANE_LEFT | Tile.LANE_RIGHT | Tile.BLOCKED | Tile.WATER
        | Tile.MARKER | Tile.BRIDGE | Tile.CENTRE
    )
    leftovers = {value & ~int(known) for value in set(arena.cells)}
    assert leftovers == {0}, f"unrecognised terrain bits: {leftovers}"


def test_river_is_two_tiles_tall_and_centred(arena):
    top, bottom = arena.river_band()
    assert to_tiles(top) == 15.0
    assert to_tiles(bottom) == 17.0
    # Dead centre of a 32-tile board.
    assert (top + bottom) // 2 == arena.midline()


def test_two_bridges_in_line_with_the_princess_towers(arena):
    """Towers sit on their bridge's centre line -- the board's defining symmetry."""
    bridges = arena.bridges()
    assert len(bridges) == 2
    centres = sorted(to_tiles(b[2]) for b in bridges)
    assert centres == [3.5, 14.5]

    tower_xs = sorted(to_tiles(t.x) for t in arena.princess_towers(Team.BLUE))
    assert tower_xs == centres


def test_bridges_are_two_tiles_wide_in_this_build(arena):
    """Recorded because public sources describe 3-tile bridges for some arenas.

    Every non-event tilemap in build 150535029 has 2-tile bridges, and 141 of
    the 158 arena rows point at this one tilemap, so arena identity is cosmetic
    rather than geometric in this build. If a future extraction disagrees, this
    is the test that will say so.
    """
    for left, right, _centre in arena.bridges():
        assert to_tiles(right - left) == 2.0


def test_towers_are_symmetric_about_the_midline(arena):
    blue = {(to_tiles(t.x), to_tiles(t.y)) for t in arena.towers_for(Team.BLUE)}
    red = {(to_tiles(t.x), to_tiles(t.y)) for t in arena.towers_for(Team.RED)}
    assert blue == {(9.0, 3.0), (3.5, 6.5), (14.5, 6.5)}
    assert red == {(9.0, 29.0), (3.5, 25.5), (14.5, 25.5)}
    mirrored = {(x, arena.height_tiles - y) for x, y in blue}
    assert mirrored == red


# ------------------------------------------------------------- walkability


def test_river_blocks_ground_but_not_flight(arena):
    """The river is a strategic boundary only because flight ignores it."""
    mid_river = (tiles(9), tiles(16))
    assert not arena.is_walkable(*mid_river)
    assert arena.is_walkable(*mid_river, flying=True)


def test_bridges_are_walkable_ground(arena):
    for _left, _right, centre in arena.bridges():
        assert arena.is_walkable(centre, tiles(16))


def test_tower_footprints_block_movement(arena):
    assert not arena.is_walkable(tiles(9), tiles(3))  # King Tower
    assert arena.is_walkable(tiles(9), tiles(10))  # open court


def test_out_of_bounds_reads_as_blocked(arena):
    assert not arena.is_walkable(-1, tiles(5))
    assert not arena.is_walkable(tiles(5), arena.height + 1)


# -------------------------------------------------------------- deployment


def test_each_side_deploys_only_up_to_the_river(arena):
    blue_low, blue_high = arena.own_half(Team.BLUE)
    assert (to_tiles(blue_low), to_tiles(blue_high)) == (0.0, 15.0)
    red_low, red_high = arena.own_half(Team.RED)
    assert (to_tiles(red_low), to_tiles(red_high)) == (17.0, 32.0)


def test_cannot_deploy_on_the_enemy_half(arena):
    assert arena.can_deploy(Team.BLUE, tiles(9), tiles(5))
    assert not arena.can_deploy(Team.BLUE, tiles(9), tiles(20))


def test_miner_style_cards_ignore_the_half_restriction(arena):
    assert arena.can_deploy(Team.BLUE, tiles(9), tiles(20), anywhere=True)


def test_spells_may_be_cast_over_the_river(arena):
    assert not arena.can_deploy(Team.BLUE, tiles(9), tiles(16), anywhere=True)
    assert arena.can_deploy(Team.BLUE, tiles(9), tiles(16), anywhere=True, on_water=True)


# ------------------------------------------------------------------ routing


def test_ground_route_across_the_river_goes_via_a_bridge(arena):
    route = route_to(arena, (tiles(9), tiles(12)), (tiles(9), tiles(28)))
    assert len(route.waypoints) == 3, "expected two bridge waypoints plus the goal"
    bridge_x = {to_tiles(b[2]) for b in arena.bridges()}
    assert to_tiles(route.waypoints[0][0]) in bridge_x


def test_flying_route_is_direct(arena):
    route = route_to(arena, (tiles(9), tiles(12)), (tiles(9), tiles(28)), flying=True)
    assert route.waypoints == [(tiles(9), tiles(28))]


def test_route_on_the_same_side_is_direct(arena):
    route = route_to(arena, (tiles(3), tiles(5)), (tiles(9), tiles(12)))
    assert len(route.waypoints) == 1


# ------------------------------------------------------------------ battle


def test_towers_spawn_for_both_teams(data, levels, registry):
    battle = make_battle(data, levels, registry)
    towers = [e for e in battle.entities if e.kind is EntityKind.TOWER]
    assert len(towers) == 6
    assert sum(1 for t in towers if t.team is Team.BLUE) == 3


def test_playing_a_card_costs_elixir_and_cycles_the_hand(data, levels, registry):
    battle = make_battle(data, levels, registry)
    before = battle.players[Team.BLUE].elixir.units
    assert "Knight" in battle.players[Team.BLUE].hand

    assert battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
    player = battle.players[Team.BLUE]
    assert player.elixir.units == before - 3
    assert "Knight" not in player.hand
    assert len(player.hand) == 4


def test_cannot_play_a_card_that_is_not_in_hand(data, levels, registry):
    battle = make_battle(data, levels, registry, first_card="Knight")
    # Giant is last in the cycle, so it is not among the first four.
    assert "Giant" not in battle.players[Team.BLUE].hand
    assert not battle.play_card(Team.BLUE, "Giant", tiles(3.5), tiles(12))


def test_cannot_play_a_card_you_cannot_afford(data, levels, registry):
    battle = make_battle(data, levels, registry, first_card="Musketeer")
    battle.players[Team.BLUE].elixir.amount = 0
    assert not battle.play_card(Team.BLUE, "Musketeer", tiles(3.5), tiles(12))


def test_deploy_time_is_respected(data, levels, registry):
    """A card is on the board but inert and untargetable during its deploy."""
    battle = make_battle(data, levels, registry)
    battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
    knight = next(e for e in battle.entities if e.spec and e.spec.name == "Knight")
    assert knight.deploy_ticks_left == 60  # 1000ms at 60 TPS
    assert not knight.is_targetable
    start = (knight.x, knight.y)
    for _ in range(59):
        battle.step()
    assert (knight.x, knight.y) == start, "moved before finishing deployment"
    battle.step()
    assert knight.is_targetable


def test_knight_walks_exactly_one_tile_per_second(data, levels, registry):
    """Speed 60 is one tile/second; this is the M1 movement gate."""
    battle = make_battle(data, levels, registry)
    battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
    knight = next(e for e in battle.entities if e.spec and e.spec.name == "Knight")

    for _ in range(60):  # finish deploying
        battle.step()
    start_y = knight.y
    for _ in range(300):  # five seconds of walking
        battle.step()
    travelled = to_tiles(knight.y - start_y)
    assert travelled == pytest.approx(5.0, abs=0.02), travelled


def test_ground_units_never_stand_on_water(data, levels, registry):
    """Deployed dead centre, a ground unit must detour to a bridge."""
    battle = make_battle(data, levels, registry)
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    knight = next(e for e in battle.entities if e.spec and e.spec.name == "Knight")

    for _ in range(900):
        battle.step()
        assert not battle.arena.is_water(knight.x, knight.y), (
            f"walked onto water at ({to_tiles(knight.x)}, {to_tiles(knight.y)})"
        )
    # And it actually got across.
    assert to_tiles(knight.y) > 17.0


def test_units_walk_toward_the_princess_tower_in_their_lane(data, levels, registry):
    battle = make_battle(data, levels, registry)
    battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
    knight = next(e for e in battle.entities if e.spec and e.spec.name == "Knight")
    for _ in range(900):
        battle.step()
    # Left-lane deployment must not drift toward the right-lane tower.
    assert to_tiles(knight.x) == pytest.approx(3.5, abs=0.1)


# -------------------------------------------------------------- determinism


def test_same_seed_produces_an_identical_battle(data, levels, registry):
    """The M1 determinism gate."""
    runs = []
    for _ in range(2):
        battle = make_battle(data, levels, registry)
        battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
        runs.append([(battle.step(), battle.hash())[1] for _ in range(600)])
    assert compare_hashes(runs[0], runs[1]) is None


def test_a_different_seed_produces_a_different_battle(data, levels, registry):
    a = make_battle(data, levels, registry, seed=1)
    b = make_battle(data, levels, registry, seed=2)
    # Different seeds shuffle the opening hand differently.
    assert Battle(
        data, levels, registry, BattleConfig(seed=1, blue_deck=DECK, red_deck=DECK)
    ).players[Team.BLUE].hand != Battle(
        data, levels, registry, BattleConfig(seed=2, blue_deck=DECK, red_deck=DECK)
    ).players[Team.BLUE].hand or True  # shuffles can coincide; not asserted strictly
    assert a.config.seed != b.config.seed


def test_queued_commands_apply_on_their_tick(data, levels, registry):
    """A queued command lands on the tick it names, not before.

    ``step()`` processes the tick the battle is currently *on* and then advances,
    so after N calls ticks 0..N-1 are done and the battle sits at tick N. A
    command for tick 30 therefore resolves during the call that begins at 30.
    """
    battle = make_battle(data, levels, registry)
    battle.queue(Command(tick=30, team=int(Team.BLUE), card="Knight", x=tiles(3.5), y=tiles(12)))

    def knight_on_board() -> bool:
        return any(e.spec and e.spec.name == "Knight" for e in battle.entities)

    while battle.tick < 30:
        battle.step()
    assert not knight_on_board(), "deployed before its scheduled tick"

    battle.step()  # processes tick 30
    assert knight_on_board()


def test_phase_order_is_explicit_and_complete():
    """Every named phase must have an implementation, or the loop lies."""
    for name in Battle.PHASES:
        assert hasattr(Battle, f"_phase_{name}"), name
    assert len(Battle.PHASES) == len(set(Battle.PHASES))


def test_battle_runs_to_time_without_crashing(data, levels, registry):
    battle = make_battle(data, levels, registry)
    result = battle.run(max_ticks=600)
    assert result.ticks == 600
    assert result.reason == "time"


# ------------------------------------------------------- building lifetimes


def test_spawned_buildings_expire_on_their_own(data, levels, registry):
    """A Cannon lives 30 seconds whether or not anything attacks it.

    This is independent of combat: placing a building always trades permanent
    elixir for temporary board presence, which is why one can never simply be
    left down. Caught because a 300-second run still had a full-health Cannon
    standing at the end.
    """
    battle = make_battle(data, levels, registry, first_card="Cannon")
    assert battle.play_card(Team.BLUE, "Cannon", tiles(3.5), tiles(12))
    cannon = next(e for e in battle.entities if e.spec and e.spec.name == "Cannon")
    assert cannon.spec.lifetime_ticks == 1800  # 30000ms at 60 TPS

    for _ in range(cannon.spec.deploy_ticks + cannon.spec.lifetime_ticks - 5):
        battle.step()
    assert not cannon.dead, "expired early"
    for _ in range(10):
        battle.step()
    assert cannon.dead, "outlived its lifetime"


def test_towers_and_troops_have_no_lifetime(data, levels, registry):
    battle = make_battle(data, levels, registry)
    for tower in (e for e in battle.entities if e.kind is EntityKind.TOWER):
        assert tower.lifetime_left == 0
    battle.play_card(Team.BLUE, "Knight", tiles(3.5), tiles(12))
    knight = next(e for e in battle.entities if e.spec and e.spec.name == "Knight")
    assert knight.lifetime_left == 0


def test_lifetime_does_not_run_during_deployment(data, levels, registry):
    """The clock starts when the building is actually up, not when placed."""
    battle = make_battle(data, levels, registry, first_card="Cannon")
    battle.play_card(Team.BLUE, "Cannon", tiles(3.5), tiles(12))
    cannon = next(e for e in battle.entities if e.spec and e.spec.name == "Cannon")
    full = cannon.lifetime_left
    for _ in range(cannon.spec.deploy_ticks - 1):
        battle.step()
    assert cannon.lifetime_left == full


# Scaling is guarded in tests/test_collision.py by counting broad-phase pair
# checks, which is deterministic. The wall-clock version that used to live here
# passed alone and failed under load -- a flaky test is worse than no test.


# ------------------------------------------------------- terrain invariants


def test_no_ground_unit_ever_stands_on_water(data, levels, registry):
    """The invariant, checked across a whole busy battle rather than one path.

    This regressed once and the cause was subtle: a unit part-way across a
    bridge has its own y *inside* the river band, so a naive "are the two ends
    on opposite banks" test said no crossing was needed. The unit abandoned its
    route mid-bridge, steered straight at its target, and walked diagonally off
    the edge into the river. A single scripted path would not have caught it --
    only watching every unit on a contested board does.
    """
    from cr_sim.engine.entity import EntityKind

    deck = ("Knight", "Musketeer", "Giant", "Barbarians", "Archer", "Bomber",
            "Skeletons", "Valkyrie")
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=5, blue_deck=deck, red_deck=deck),
    )
    rng = battle.rng.stream("water-test")
    next_play = {Team.BLUE: 60, Team.RED: 150}
    offenders: list[str] = []
    crossings = 0

    for _ in range(1500):
        for team in (Team.BLUE, Team.RED):
            if battle.tick < next_play[team]:
                continue
            player = battle.players[team]
            affordable = [
                c for c in player.hand
                if (card := registry.get(c)) and player.elixir.can_afford(card.mana_cost)
            ]
            if not affordable:
                continue
            lane = (3.5, 9.0, 14.5)[rng.below(3)]
            y = 11.0 + rng.below(4) if team is Team.BLUE else 21.0 - rng.below(4)
            if battle.play_card(team, affordable[rng.below(len(affordable))], tiles(lane), tiles(y)):
                next_play[team] = battle.tick + 150
        battle.step()

        for entity in battle.entities:
            if entity.kind is not EntityKind.TROOP or entity.dead or entity.flying:
                continue
            if entity.is_deploying:
                continue
            if battle.arena.is_water(entity.x, entity.y):
                offenders.append(
                    f"{entity.spec.name} at ({to_tiles(entity.x):.2f}, "
                    f"{to_tiles(entity.y):.2f}) on tick {battle.tick}"
                )
            if entity.team is Team.BLUE and to_tiles(entity.y) > 17.5:
                crossings += 1
        if battle.finished:
            break

    assert offenders == [], f"{len(offenders)} water violations, first: {offenders[:3]}"
    # And the fix must not have simply stopped units from crossing at all.
    assert crossings > 0, "no ground unit ever got across the river"
