"""M3 gate: collision, pushback and the spatial index.

Collision is what turns a card into an *area* of board control. Without it a
Skeleton Army is fifteen units sharing one point, a swarm files across a bridge
single-file, and surrounding a Prince does nothing. The tests here are about
that behaviour, plus the two invariants it must never break: units stay out of
terrain, and the result stays deterministic.
"""

from __future__ import annotations

import itertools
import time

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import distance, pack_offsets, tiles, to_tiles
from cr_sim.engine.movement import IMMOVABLE_MASS, separate
from cr_sim.engine.spatial import SpatialIndex
from cr_sim.engine.specs import build_unit_spec
from cr_sim.replay import compare_hashes

from .test_data_pipeline import BUILD

RARITY = {
    "Knight": "Common", "Skeleton": "Common", "Golem": "Epic", "Giant": "Rare",
    "Pekka": "Epic", "Barbarian": "Common", "Minion": "Common", "Musketeer": "Rare",
}


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, *, seed=1, deck=("Skeletons",) * 8):
    data, levels, registry = world
    return Battle(data, levels, registry, BattleConfig(seed=seed, blue_deck=deck, red_deck=deck))


def _spawn(battle, world, name, team, x, y):
    data, levels, _registry = world
    rarity = RARITY.get(name, "Common")
    scale = levels.get(rarity)
    spec = build_unit_spec(
        data, levels, name, level=scale.internal_level(11), rarity=rarity, clock=battle.clock
    )
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y), hitpoints=spec.hitpoints,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    battle._register(entity)
    return entity


def _overlapping(entities) -> int:
    count = 0
    for a, b in itertools.combinations(entities, 2):
        if a.flying != b.flying:
            continue
        gap = distance(a.x, a.y, b.x, b.y)
        if gap < a.collision_radius + b.collision_radius - 100:
            count += 1
    return count


# --------------------------------------------------------------- separation


def test_a_swarm_stays_an_area_not_a_point(world):
    """The whole point of collision: fifteen units occupy fifteen units of space.

    Some mutual compression is expected and correct -- they are all walking at
    the same tower, and a crowd converging on one point squeezes in this game
    too. What matters is that the compression is *bounded* rather than
    collapsing back to a single stack.
    """
    battle = _battle(world, deck=("SkeletonArmy",) * 8)
    battle.players[Team.BLUE].elixir.add(10)
    battle.play_card(Team.BLUE, "SkeletonArmy", tiles(9), tiles(10))
    skeletons = [e for e in battle.entities if e.spec and e.spec.name == "Skeleton"]
    assert len(skeletons) == 15

    for _ in range(300):
        battle.step()
    worst = max(
        (a.collision_radius + b.collision_radius) - distance(a.x, a.y, b.x, b.y)
        for a, b in itertools.combinations(skeletons, 2)
    )
    assert to_tiles(worst) < 0.15, f"crowd compressed {to_tiles(worst):.3f} tiles"
    # And they still cover real ground rather than stacking.
    xs = [e.x for e in skeletons]
    ys = [e.y for e in skeletons]
    assert to_tiles(max(xs) - min(xs)) > 1.5
    assert to_tiles(max(ys) - min(ys)) > 1.0


def test_a_light_unit_yields_to_a_heavy_one(world):
    """Mass decides who moves; a Skeleton bounces off a Golem, not the reverse."""
    battle = _battle(world)
    golem = _spawn(battle, world, "Golem", Team.BLUE, 9, 10.0)
    skeleton = _spawn(battle, world, "Skeleton", Team.BLUE, 9, 10.2)
    golem_start = (golem.x, golem.y)
    skeleton_start = (skeleton.x, skeleton.y)

    for _ in range(30):
        battle.step()

    golem_moved = distance(*golem_start, golem.x, golem.y)
    skeleton_moved = distance(*skeleton_start, skeleton.x, skeleton.y)
    assert skeleton_moved > golem_moved


def test_ignore_pushback_units_are_never_displaced(world):
    """Not merely heavy -- immovable. A committed Prince cannot be shoved aside.

    Towers are removed so the P.E.K.K.A has nowhere to walk: what is measured is
    displacement *by the crowd*, not its own movement.
    """
    battle = _battle(world)
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}

    pekka = _spawn(battle, world, "Pekka", Team.BLUE, 9, 10.0)
    assert pekka.spec.ignore_pushback
    crowd = [_spawn(battle, world, "Skeleton", Team.RED, 9, 10.0 + 0.05 * i) for i in range(6)]
    start = (pekka.x, pekka.y)

    for _ in range(60):
        battle.step()
    assert (pekka.x, pekka.y) == start, "an IgnorePushback unit was displaced"
    # The crowd, by contrast, is very much displaced.
    assert any((e.x, e.y) != (tiles(9), tiles(10.0 + 0.05 * i)) for i, e in enumerate(crowd))


def test_buildings_cannot_be_pushed(world):
    battle = _battle(world)
    tower = next(e for e in battle._towers[Team.BLUE] if "King" not in e.spec.name)
    start = (tower.x, tower.y)
    for i in range(8):
        _spawn(battle, world, "Skeleton", Team.BLUE, to_tiles(tower.x), to_tiles(tower.y) + 0.1 * i)
    for _ in range(60):
        battle.step()
    assert (tower.x, tower.y) == start


def test_air_and_ground_do_not_collide(world):
    """They occupy different layers; a Minion sits happily over a Knight."""
    battle = _battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    minion = _spawn(battle, world, "Minion", Team.BLUE, 9, 10.0)
    assert minion.flying and not knight.flying
    for _ in range(30):
        battle.step()
    assert distance(knight.x, knight.y, minion.x, minion.y) < knight.collision_radius


def test_separation_never_pushes_a_unit_into_the_river(world):
    """Terrain wins over separation, or crowds squeeze units into the water."""
    battle = _battle(world)
    crowd = [
        _spawn(battle, world, "Skeleton", Team.BLUE, 3.5, 14.6 + 0.05 * i) for i in range(10)
    ]
    for _ in range(120):
        battle.step()
        for unit in crowd:
            assert not battle.arena.is_water(unit.x, unit.y), (
                f"pushed onto water at ({to_tiles(unit.x):.2f}, {to_tiles(unit.y):.2f})"
            )


def test_two_immovable_units_are_left_alone(world):
    """Nothing to resolve, and pretending otherwise would move a tower."""
    battle = _battle(world)
    a = _spawn(battle, world, "Golem", Team.BLUE, 9, 10.0)
    b = _spawn(battle, world, "Pekka", Team.BLUE, 9, 10.1)
    before = ((a.x, a.y), (b.x, b.y))
    assert separate(a, b, battle.arena) is False
    assert ((a.x, a.y), (b.x, b.y)) == before


def test_coincident_units_still_separate(world):
    """Exactly overlapping units have no direction to push along; pick one."""
    battle = _battle(world)
    a = _spawn(battle, world, "Skeleton", Team.BLUE, 9, 10.0)
    b = _spawn(battle, world, "Skeleton", Team.BLUE, 9, 10.0)
    assert (a.x, a.y) == (b.x, b.y)
    assert separate(a, b, battle.arena)
    assert (a.x, a.y) != (b.x, b.y)


# ------------------------------------------------------------------ packing


def test_spawn_packing_leaves_no_overlap():
    """Derived layout for cards that ship no SummonRadius."""
    radius = 9000  # half a tile
    for count in (2, 3, 5, 8, 15, 30):
        offsets = pack_offsets(count, radius)
        assert len(offsets) == count
        for (ax, ay), (bx, by) in itertools.combinations(offsets, 2):
            assert distance(ax, ay, bx, by) >= 2 * radius - 200, count


def test_skeleton_army_spawns_without_overlap(world):
    """Fifteen units, and the card specifies no radius at all."""
    battle = _battle(world, deck=("SkeletonArmy",) * 8)
    battle.players[Team.BLUE].elixir.add(10)
    battle.play_card(Team.BLUE, "SkeletonArmy", tiles(9), tiles(10))
    skeletons = [e for e in battle.entities if e.spec and e.spec.name == "Skeleton"]
    assert _overlapping(skeletons) == 0


# ------------------------------------------------------------ spatial index


def test_index_finds_everything_a_full_scan_would():
    """The broad phase may over-return, but must never miss a neighbour."""
    from cr_sim.engine.entity import reset_entity_ids

    reset_entity_ids()
    entities = [
        Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=tiles(x), y=tiles(y), hitpoints=1)
        for x in range(0, 18, 2)
        for y in range(0, 32, 3)
    ]
    index = SpatialIndex(tiles(18), tiles(32))
    index.rebuild(entities)
    assert len(index) == len(entities)

    probe = entities[10]
    reach = tiles(3)
    found = {e.id for e in index.candidates(probe, reach)}
    expected = {
        e.id
        for e in entities
        if e is not probe and distance(probe.x, probe.y, e.x, e.y) <= reach
    }
    assert expected <= found, "the index missed a genuine neighbour"


def test_index_yields_each_pair_once():
    from cr_sim.engine.entity import reset_entity_ids

    reset_entity_ids()
    entities = [
        Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=tiles(9), y=tiles(10 + 0.2 * i),
               hitpoints=1, collision_radius=tiles(0.5))
        for i in range(5)
    ]
    index = SpatialIndex(tiles(18), tiles(32))
    index.rebuild(entities)
    pairs = list(index.pairs(tiles(0.5)))
    seen = {tuple(sorted((a.id, b.id))) for a, b in pairs}
    assert len(seen) == len(pairs), "a pair was yielded twice"


def test_index_ignores_the_dead():
    from cr_sim.engine.entity import reset_entity_ids

    reset_entity_ids()
    alive = Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=tiles(9), y=tiles(10), hitpoints=1)
    dead = Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=tiles(9), y=tiles(10), hitpoints=1)
    dead.dead = True
    index = SpatialIndex(tiles(18), tiles(32))
    index.rebuild([alive, dead])
    assert len(index) == 1


# ----------------------------------------------------------- still correct


def test_collision_preserves_determinism(world):
    """Separation order must not vary between runs."""
    runs = []
    for _ in range(2):
        battle = _battle(world, deck=("SkeletonArmy",) * 8)
        battle.players[Team.BLUE].elixir.add(10)
        battle.play_card(Team.BLUE, "SkeletonArmy", tiles(9), tiles(12))
        runs.append([(battle.step(), battle.hash())[1] for _ in range(400)])
    assert compare_hashes(runs[0], runs[1]) is None


def test_dead_units_leave_the_live_list(world):
    """Corpses stop costing a scan in every phase for the rest of the match."""
    battle = _battle(world)
    victim = _spawn(battle, world, "Skeleton", Team.BLUE, 9, 10.0)
    assert victim in battle.entities
    victim.kill()
    battle.step()
    assert victim not in battle.entities
    assert victim in battle.graveyard
    # ...but stays reachable, so a stale target reference resolves to a corpse.
    assert battle._entity(victim.id) is victim


def test_broad_phase_does_far_less_work_than_a_full_sweep(world):
    """The index must cost a small fraction of comparing everything to everything.

    Note what is *not* claimed: that pairs-per-unit stays constant. On a fixed
    board more units means proportionally higher density, so each unit really
    does have more neighbours, and that growth is physical rather than a defect
    of the index. What the index buys is the constant: a unit is compared
    against its own neighbourhood instead of against the entire board.
    """
    def measure(units: int) -> tuple[float, float]:
        battle = _battle(world)
        battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
        battle._towers = {Team.BLUE: [], Team.RED: []}
        for i in range(units):
            _spawn(battle, world, "Skeleton", Team.BLUE, 1 + (i * 7 % 16), 1 + (i * 5 % 13))
        battle._index.rebuild(battle.entities)
        indexed = sum(1 for _ in battle._index.pairs(battle._max_radius))
        naive = units * (units - 1) / 2
        return indexed, naive

    for units in (40, 120, 200):
        indexed, naive = measure(units)
        assert indexed < naive * 0.25, (
            f"{units} units: index checked {indexed:.0f} pairs of a possible {naive:.0f}"
        )
    # Measured ratios are around a tenth: 0.077 at 40 units, 0.096 at 200. The
    # ratio drifts *up* slightly with density rather than down, which is the
    # honest shape -- a fuller board genuinely has more real neighbours.


def _reference_sweep(index, arena, *, max_radius, passes=3):
    """The collision sweep written the obvious way, as an oracle.

    Straight ``SpatialIndex.pairs`` with the rejects applied to each pair it
    yields, and no attempt to skip anything. The shipping version folds those
    rejects into its own copy of the broad-phase walk and skips, in later
    passes, whatever cannot have changed -- both worth real throughput, and
    both exactly the kind of thing that drifts. This is what it has to agree
    with.
    """
    from cr_sim.engine.movement import _INCORPOREAL, separate as _separate

    moved = 0
    for _ in range(passes):
        touched = 0
        for a, b in index.pairs(max_radius):
            if a.dead or b.dead:
                continue
            if a.kind in _INCORPOREAL or b.kind in _INCORPOREAL:
                continue
            if a.flying != b.flying:
                continue
            if a.deploy_ticks_left > 0 or b.deploy_ticks_left > 0:
                continue
            reach = a.collision_radius + b.collision_radius
            if reach <= 0:
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            if dx * dx + dy * dy >= reach * reach:
                continue
            if _separate(a, b, arena):
                touched += 1
        moved += touched
        if not touched:
            break
    return moved


def test_the_collision_sweep_matches_the_obvious_implementation(world):
    """The optimised sweep must land every unit exactly where the plain one does.

    Two shortcuts are being pinned here. The sweep repeats the broad-phase walk
    rather than consuming it, so that a candidate it is going to reject never
    has to exist as a tuple; and its later relaxation passes skip any entity
    whose whole search span has stood still, because a pair neither of whose
    ends has moved must reach the verdict it reached last pass.

    Compared by outcome rather than by call sequence, which is the claim that
    actually matters: separating one pair moves the units the next pair is
    measured against, so agreeing on the final positions means agreeing on the
    order as well. Checked from several points in a match, because a shortcut
    that only shows up on a crowded board is exactly the shortcut that would
    slip through a single sparse fixture.
    """
    from cr_sim.engine.movement import resolve_collisions

    battle = _battle(world, deck=("SkeletonArmy",) * 8)
    for team in (Team.BLUE, Team.RED):
        battle.players[team].elixir.add(10)
    battle.play_card(Team.BLUE, "SkeletonArmy", tiles(9), tiles(15))
    battle.play_card(Team.RED, "SkeletonArmy", tiles(9), tiles(17))

    compared = 0
    for tick in range(160):
        battle.step()
        if tick % 20 or tick < 40:
            continue
        # Two identical boards; one gets the oracle, the other the real thing.
        plain, fast = battle.clone(), battle.clone()
        plain._index.rebuild(plain.entities)
        fast._index.rebuild(fast.entities)
        expected = _reference_sweep(
            plain._index, plain.arena, max_radius=plain._max_radius
        )
        actual = resolve_collisions(
            fast._index, fast.arena, max_radius=fast._max_radius
        )
        assert actual == expected, f"tick {tick}: separation count differs"
        assert [(e.id, e.x, e.y) for e in plain.entities] == [
            (e.id, e.x, e.y) for e in fast.entities
        ], f"tick {tick}: units ended up somewhere else"
        if expected:
            compared += 1

    assert compared, "no fixture tick actually had anything overlapping"
