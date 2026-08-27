"""M6: what a unit leaves behind, and what a building produces.

Two mechanics that decide whether the expensive cards are the cards they are.
A Golem that does not split is a worse Giant, and killing one would end a push
rather than begin the second half of it. A Tombstone that does not produce
Skeletons is a 3-elixir building with 400 hitpoints and no purpose.

The numbers asserted here come from the build. Where a value corroborates a
figure players know -- Witch's seven-second cycle, the Giant Skeleton's
three-second fuse -- that is said in the test, because those are what pin the
reading of the field down.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, card):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=("Knight",) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _dummy(battle, world, x, y, *, team=Team.RED, hitpoints=99_999):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "Knight", level=11, rarity="Common", clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y), hitpoints=hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = hitpoints
    entity.deploy_ticks_left = 0
    battle._register(entity)
    return entity


def _deploy_and_kill(battle, card, *, settle=200):
    """Put the card down, let it land, then kill it outright."""
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(12)), card
    for _ in range(settle):
        battle.step()
    unit = next(
        e for e in battle.entities
        if e.team is Team.BLUE and e.kind in (EntityKind.TROOP, EntityKind.BUILDING)
    )
    unit.hitpoints = 1
    unit.apply_damage(1)
    return unit


def _living(battle, name):
    return [e for e in battle.entities if not e.dead and e.spec is not None and e.spec.name == name]


# ------------------------------------------------------------- death spawns


@pytest.mark.parametrize(
    "card,child,count",
    [
        ("Golem", "Golemite", 2),
        ("LavaHound", "LavaPups", 6),
        ("Tombstone", "Skeleton", 4),
        ("BattleRam", "Barbarian", 2),
        ("ElixirGolem", "ElixirGolem2", 2),
    ],
)
def test_a_death_spawn_produces_its_children(world, card, child, count):
    battle = _battle(world, card)
    _deploy_and_kill(battle, card)
    for _ in range(180):
        battle.step()
    assert len(_living(battle, child)) == count


def test_death_spawns_are_spread_rather_than_stacked(world):
    """Units on one exact point are perfectly overlapped, so one Zap kills all four.

    Spacing them is not cosmetic: it is the difference between a Tombstone's
    Skeletons being a real obstacle and being a single splash-shaped target.
    """
    battle = _battle(world, "Tombstone")
    _deploy_and_kill(battle, "Tombstone")
    for _ in range(180):
        battle.step()
    spots = {(e.x, e.y) for e in _living(battle, "Skeleton")}
    assert len(spots) == 4, "the death spawn stacked on one point"


def test_a_golem_explodes_when_it_dies(world):
    """88 base in a 2-tile radius, which is a real cost of playing one."""
    battle = _battle(world, "Golem")
    victim = _dummy(battle, world, 9, 12.8)
    _deploy_and_kill(battle, "Golem")
    for _ in range(30):
        battle.step()
    assert victim.hitpoints < victim.max_hitpoints, "no death blast"


def test_a_death_blast_hits_its_own_side_too(world):
    """Dropping a Golem on your own support costs you the support.

    Filtering the blast to enemies would quietly make every big death spawn
    strictly better than the card actually is.
    """
    battle = _battle(world, "Golem")
    friend = _dummy(battle, world, 9, 12.8, team=Team.BLUE)
    _deploy_and_kill(battle, "Golem")
    for _ in range(30):
        battle.step()
    assert friend.hitpoints < friend.max_hitpoints


def test_death_pushback_shoves_survivors_clear(world):
    battle = _battle(world, "Golem")
    victim = _dummy(battle, world, 9, 12.8)
    start = victim.y
    _deploy_and_kill(battle, "Golem")
    for _ in range(30):
        battle.step()
    assert victim.y != start, "death pushback did nothing"


# -------------------------------------------------------------------- fuses


def test_the_giant_skeletons_bomb_detonates_after_three_seconds(world):
    """DeployTime 3000 on a hitpoint-less building is the famous fuse.

    Read as an ordinary deploy timer the bomb would sit inert forever; read as
    a fuse it is the whole card.
    """
    battle = _battle(world, "GiantSkeleton")
    victim = _dummy(battle, world, 9, 12.5)
    _deploy_and_kill(battle, "GiantSkeleton")

    before = victim.hitpoints
    fired = None
    for tick in range(400):
        battle.step()
        if victim.hitpoints != before:
            fired = tick
            break
    assert fired is not None, "the bomb never went off"
    assert 175 <= fired <= 185, f"fuse was {fired / 60:.2f}s, expected 3.00s"


def test_a_bomb_cannot_be_shot_down(world):
    """Nothing in the game destroys a live bomb, and letting it be targeted
    would also let it be killed early, cancelling the blast."""
    battle = _battle(world, "GiantSkeleton")
    _deploy_and_kill(battle, "GiantSkeleton")
    for _ in range(20):
        battle.step()
    bombs = [e for e in battle.entities if e.spec is not None and e.spec.is_fuse]
    assert bombs, "no bomb was spawned"
    assert not any(b.is_targetable for b in bombs)


# ----------------------------------------------------------------- spawners


def test_a_witch_produces_four_skeletons_every_seven_seconds(world):
    """7000ms is what pins SpawnPauseTime down as the wave interval.

    It is the cycle the card is documented for, and no other field in the row
    produces it -- Witch carries no SpawnInterval at all.
    """
    battle = _battle(world, "Witch")
    assert battle.play_card(Team.BLUE, "Witch", tiles(9), tiles(12))

    arrivals, previous = [], 0
    for tick in range(60 * 30):
        battle.step()
        living = len(_living(battle, "Skeleton"))
        if living > previous:
            arrivals.append(tick)
        previous = living

    assert arrivals, "the Witch never spawned"
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert gaps, "only one wave in thirty seconds"
    for gap in gaps:
        assert 415 <= gap <= 425, f"wave gap {gap / 60:.2f}s, expected 7.00s"


def test_a_tombstone_spawns_on_its_own_shorter_cycle(world):
    battle = _battle(world, "Tombstone")
    assert battle.play_card(Team.BLUE, "Tombstone", tiles(9), tiles(12))
    arrivals, previous = [], 0
    for tick in range(60 * 20):
        battle.step()
        living = len(_living(battle, "Skeleton"))
        if living > previous:
            arrivals.append(tick)
        previous = living
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert gaps and all(200 <= g <= 220 for g in gaps), f"gaps were {gaps}"


def test_a_spawner_produces_nothing_while_it_is_still_landing(world):
    """A hut that has not finished deploying is not yet a hut."""
    battle = _battle(world, "Witch")
    battle.play_card(Team.BLUE, "Witch", tiles(9), tiles(12))
    witch = next(e for e in battle.entities if e.team is Team.BLUE and e.kind is EntityKind.TROOP)
    assert witch.is_deploying
    while witch.is_deploying:
        battle.step()
        assert not _living(battle, "Skeleton"), "spawned mid-deploy"


# ------------------------------------------------------------ legal ground


def test_a_death_spawn_never_lands_in_the_river(world):
    """Movement has always guarded this; spawning never did.

    Every offset that places a unit relative to a point can push it off legal
    ground -- a swarm's ring, a death spawn's radius, a Graveyard skeleton's
    annulus. A soak run found troops standing in the river in 2.5% of matches.
    """
    battle = _battle(world, "Tombstone")
    top, bottom = battle.arena.river_band()
    # Killed on the near bank, so the spawn ring reaches across the water.
    battle._spawn_units(
        team=Team.BLUE, character="Skeleton", count=8,
        x=tiles(9), y=top + (bottom - top) // 2, radius=tiles(2), rarity="Common",
    )
    for entity in battle.entities:
        if entity.spec is not None and entity.spec.name == "Skeleton":
            assert battle.arena.is_walkable(entity.x, entity.y, flying=False), (
                f"skeleton spawned on impassable ground at "
                f"({to_tiles(entity.x):.2f}, {to_tiles(entity.y):.2f})"
            )


def test_settling_leaves_a_legal_point_alone(world):
    """It is a nudge, not a snap. A unit slightly out of place is a smaller
    wrong than one teleported across the board."""
    battle = _battle(world, "Tombstone")
    point = (tiles(9), tiles(10))
    assert battle._settle(*point, flying=False) == point


def test_a_flying_unit_is_settled_against_its_own_rules(world):
    """The river is walkable for something with wings."""
    battle = _battle(world, "Tombstone")
    top, bottom = battle.arena.river_band()
    over_water = (tiles(9), top + (bottom - top) // 2)
    assert battle._settle(*over_water, flying=True) == over_water


# --------------------------------------------------------- riders, not huts


@pytest.mark.parametrize(
    "card, rider, expected",
    [("GoblinGiant", "SpearGoblinGiant", 2), ("RamRider", "RamRider", 1)],
)
def test_a_carried_rider_is_put_down_once_and_not_on_a_loop(world, card, rider, expected):
    """No ``SpawnPauseTime`` means one wave, not an infinitely fast one.

    Four entities in the build set ``SpawnCharacter`` with no pause time, and
    every one of them also sets ``SpawnAttach``: they are riders, not spawners.
    Goblin Giant carries two Spear Goblins on its back and the Ram carries its
    rider, each put down once when the carrier lands.

    Falling back to a one-tick period turned that into a wave *every tick*. A
    single Goblin Giant produced 242 Spear Goblins in five seconds and a Ram
    Rider 121 riders -- not a rounding error in a card, a different game.
    """
    battle = _battle(world, card)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(8))
    for _ in range(600):
        battle.step()

    carried = [
        e for e in battle.entities
        if not e.dead and e.spec is not None and e.spec.name == rider
        and e.team is Team.BLUE
    ]
    assert len(carried) == expected, f"{len(carried)} {rider}s from one {card}"


def test_a_hut_still_produces_on_its_cycle(world):
    """The guard above must not stop the buildings that genuinely repeat.

    Witch's ``SpawnPauseTime`` is 7000ms for four Skeletons, which is the
    seven-second cycle the card is known for.
    """
    battle = _battle(world, "Witch")
    assert battle.play_card(Team.BLUE, "Witch", tiles(9), tiles(8))
    for _ in range(700):
        battle.step()
    skeletons = [
        e for e in battle.entities + battle.graveyard
        if e.spec is not None and e.spec.name == "Skeleton" and e.team is Team.BLUE
    ]
    assert len(skeletons) >= 8, f"{len(skeletons)} skeletons in 11 seconds"


def test_a_death_spawn_projectile_detonates_where_the_body_fell(world):
    """``DeathSpawnProjectile`` is a separate column from ``DeathDamage``.

    Five entities use it and none of them also carries ``DeathDamage``, so for
    every one of them it is the *only* death payload there is. Goblin
    Demolisher is the clearest: neither its ordinary form nor its kamikaze form
    has a ``Damage`` column at all, so unread, a suicide unit charged a
    building and did nothing whatsoever to it. Phoenix and Goblin Party Hut
    carry the same field.
    """
    battle = _battle(world, "GoblinDemolisher")
    demolisher = battle._spawn_units(
        team=Team.BLUE, character="GoblinDemolisher", count=1,
        x=tiles(9), y=tiles(12),
    )[0]
    victim = battle._spawn_units(
        team=Team.RED, character="Golem", count=1,
        x=tiles(9), y=tiles(12.5), rarity="Epic",
    )[0]
    demolisher.deploy_ticks_left = 0
    victim.deploy_ticks_left = 0

    before = victim.hitpoints
    demolisher.kill()
    for _ in range(60):
        battle.step()
    assert victim.hitpoints < before, "the Demolisher's dynamite did nothing"
