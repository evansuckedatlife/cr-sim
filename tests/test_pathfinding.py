"""Weighted pathfinding.

Until this existed a building was decorative: a Cannon dropped in the lane to
pull a push off its line pulled nothing, because nothing in the movement code
ever considered going around it. Units walked at their target in a straight
line and only ever planned a route to cross the river.

The costs come from the build, and they say what the mechanic is. A building
is 50 against a default of 8 -- expensive, deliberately not impassable, so a
unit with nowhere else to go still gets through. Nearly every test here exists
because some plausible-looking implementation gets that distinction wrong in
one direction or the other.
"""

from __future__ import annotations

import pytest

from cr_sim.data.source import LogicData
from cr_sim.engine.arena import load_arena
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.pathgrid import (
    PATH_COSTS,
    PathGrid,
    field_path,
    find_path,
    flow_field,
    load_path_costs,
    next_cell,
    simplify,
)
from cr_sim.engine.pathing import line_blocked, route_to

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, load_arena(data)


@pytest.fixture
def grid(world):
    data, arena = world
    return PathGrid(arena, load_path_costs(data.globals_map()))


def _wall(grid, xs, ys, cost=50):
    return {grid.index_of(cx, cy): cost for cx in xs for cy in ys}


# ------------------------------------------------------------------- costs


def test_costs_come_from_the_build(world):
    """Read, not invented, so a rebalance in the files is picked up."""
    data, _ = world
    costs = load_path_costs(data.globals_map())
    assert costs["default"] == 8
    assert costs["road"] == 5
    assert costs["building"] == 50
    assert costs["building"] > costs["default"] * 5, "a building must be a real detour"


def test_missing_globals_fall_back_rather_than_crashing():
    assert load_path_costs({}) == PATH_COSTS


def test_ground_cannot_enter_the_river_but_air_can(grid):
    """Not a cost -- an exclusion.

    A ground unit does not cross the river slowly, it does not cross it, and a
    cost model that let it would produce paths no unit can walk.
    """
    water = [
        (cx, cy)
        for cy in range(grid.arena.half_height)
        for cx in range(grid.arena.half_width)
        if grid.cost(cx, cy, flying=False) == 0 and grid.cost(cx, cy, flying=True) > 0
    ]
    assert water, "no cell is impassable on the ground and passable in the air"


def test_a_bridge_is_the_cheapest_ground_on_the_board(grid):
    """Which is why every ground push funnels onto one."""
    costs = grid.costs
    bridge_cells = []
    top, bottom = grid.arena.river_band()
    half = tiles(0.5)
    for cy in range(top // half, bottom // half + 1):
        for cx in range(grid.arena.half_width):
            cost = grid.cost(cx, cy)
            if cost:
                bridge_cells.append(cost)
    assert bridge_cells, "the river band has no walkable cells at all"
    assert min(bridge_cells) <= costs["road"] < costs["default"]


# ---------------------------------------------------------------- the search


def test_a_clear_lane_is_a_straight_line(grid):
    path = find_path(grid, (7, 20), (7, 28))
    assert {c[0] for c in path} == {7}, "wandered off a clear straight line"


def test_a_building_is_routed_around(grid):
    grid.set_occupancy(_wall(grid, range(5, 10), range(23, 26)))
    path = find_path(grid, (7, 20), (7, 28))
    assert path, "no route at all past a building"
    assert not any(grid.index_of(*c) in _wall(grid, range(5, 10), range(23, 26)) for c in path)
    assert max(c[0] for c in path) > 7 or min(c[0] for c in path) < 7, "did not deviate"


def test_a_building_is_pushed_through_when_there_is_no_way_around(grid):
    """The reason it is a cost and not a wall.

    A unit boxed in by buildings has to be able to leave, and in the game it
    does -- it attacks its way out. A hard obstacle would strand it.
    """
    grid.set_occupancy(_wall(grid, range(grid.arena.half_width), range(23, 25)))
    path = find_path(grid, (7, 20), (7, 28))
    assert path, "a full-width wall of buildings made the board impassable"


def test_no_route_returns_empty_rather_than_raising(grid):
    """A unit with nowhere to go falls back to walking at its target."""
    assert find_path(grid, (7, 20), (-5, -5)) == []


def test_a_diagonal_never_cuts_between_two_impassable_cells(grid):
    """Or a unit slips through a gap the geometry does not have.

    Asserted over a real path across the river rather than a contrived one:
    occupancy can only make a cell *expensive*, never impassable, so the only
    way to exercise this is against terrain that genuinely is -- which the
    riverbank has plenty of.
    """
    path = find_path(grid, (7, 10), (28, 54))
    assert len(path) > 10, "no path to check"
    for before, after in zip(path, path[1:]):
        dx, dy = after[0] - before[0], after[1] - before[1]
        if dx and dy:
            assert grid.cost(before[0] + dx, before[1]) > 0
            assert grid.cost(before[0], before[1] + dy) > 0


# ------------------------------------------------------------- flow fields


def test_a_field_gives_the_same_endpoint_as_the_search(grid):
    """The field is an optimisation, not a different answer."""
    goal = (28, 54)
    assert find_path(grid, (7, 10), goal)[-1] == field_path(grid, (7, 10), goal)[-1] == goal


def test_a_field_is_cached_until_occupancy_changes(grid):
    goal = (28, 54)
    flow_field(grid, goal)
    assert grid._fields, "nothing was cached"
    before = grid.version
    grid.set_occupancy(_wall(grid, range(5, 8), range(30, 32)))
    assert grid.version > before
    assert not grid._fields, "a stale field survived a change to the board"


def test_an_unchanged_occupancy_does_not_invalidate(grid):
    """The version invalidates every cached field, so one that moved every
    tick would switch the cache off entirely."""
    wall = _wall(grid, range(5, 8), range(30, 32))
    grid.set_occupancy(wall)
    flow_field(grid, (28, 54))
    version = grid.version
    grid.set_occupancy(dict(wall))
    assert grid.version == version and grid._fields


def test_a_field_step_always_reduces_the_remaining_cost(grid):
    goal = (7, 40)
    field = flow_field(grid, goal)
    cell = (7, 20)
    width = grid.arena.half_width
    for _ in range(200):
        step = next_cell(grid, field, cell)
        if step is None:
            break
        assert field[step[1] * width + step[0]] < field[cell[1] * width + cell[0]]
        cell = step
    assert cell == goal, "the field did not lead to its goal"


# ------------------------------------------------------------------ routes


def test_a_clear_route_is_one_waypoint(world, grid):
    _data, arena = world
    route = route_to(arena, (tiles(3.5), tiles(17)), (tiles(3.5), tiles(24)), grid=grid)
    assert len(route.waypoints) == 1


def test_a_building_wall_bends_the_route(world, grid):
    _data, arena = world
    grid.set_occupancy(_wall(grid, range(3, 12), range(38, 42)))
    route = route_to(arena, (tiles(3.5), tiles(17)), (tiles(3.5), tiles(24)), grid=grid)
    xs = [to_tiles(x) for x, _ in route.waypoints]
    assert max(abs(x - 3.5) for x in xs) > 1.0, "the route ignored the wall"


def test_a_route_still_ends_where_it_was_asked_to(world, grid):
    """The field lands on a cell centre; the caller asked for a point."""
    _data, arena = world
    grid.set_occupancy(_wall(grid, range(3, 12), range(38, 42)))
    goal = (tiles(3.5), tiles(24))
    assert route_to(arena, (tiles(3.5), tiles(17)), goal, grid=grid).waypoints[-1] == goal


def test_flying_units_ignore_all_of_it(world, grid):
    _data, arena = world
    grid.set_occupancy(_wall(grid, range(0, 36), range(30, 40)))
    goal = (tiles(9), tiles(28))
    route = route_to(arena, (tiles(9), tiles(4)), goal, flying=True, grid=grid)
    assert route.waypoints == [goal], "a flying unit was routed around something"


# ------------------------------------------------------------ when to path


def test_expensive_counts_as_blocked(grid):
    """The distinction that made the first version do nothing.

    A building is passable at cost 50, so a check that only looked for
    impassable cells never fired for the one obstacle the whole feature exists
    to handle.
    """
    start, goal = (tiles(3.5), tiles(17)), (tiles(3.5), tiles(24))
    assert not line_blocked(grid, start, goal)
    grid.set_occupancy(_wall(grid, range(3, 12), range(38, 42)))
    assert line_blocked(grid, start, goal)


def test_flying_is_never_blocked(grid):
    grid.set_occupancy(_wall(grid, range(0, 36), range(30, 40)))
    assert not line_blocked(
        grid, (tiles(9), tiles(4)), (tiles(9), tiles(28)), flying=True
    )


def test_simplify_keeps_only_the_corners(grid):
    straight = [(7, y) for y in range(20, 30)]
    assert simplify(straight) == [(7, 20), (7, 29)]
    bent = [(7, 20), (7, 21), (8, 22), (9, 23)]
    assert len(simplify(bent)) < len(bent)
